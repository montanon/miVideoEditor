"""Specialized storage service for annotation operations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mivideoeditor.core.models import SensitiveArea
from mivideoeditor.storage.file_manager import FileManager
from mivideoeditor.storage.models import AnnotationRecord
from mivideoeditor.storage.service import StorageService

logger = logging.getLogger(__name__)


class AnnotationService:
    """High-level service for annotation storage and retrieval operations."""

    def __init__(self, storage_service: StorageService, file_manager: FileManager):
        self.storage = storage_service
        self.file_manager = file_manager

    def save_annotation(
        self, annotation: SensitiveArea, frame_image: np.ndarray | None = None
    ) -> str:
        """Save annotation with optional frame image."""
        # Store frame image if provided
        image_path = None
        if frame_image is not None:
            image_path = self.file_manager.store_frame_image(
                frame_image, annotation.id, annotation.timestamp, optimize=True
            )

        # Convert SensitiveArea to AnnotationRecord
        annotation_record = AnnotationRecord(
            id=annotation.id,
            video_id=annotation.metadata.get("video_id", ""),
            timestamp=annotation.timestamp,
            frame_number=annotation.metadata.get("frame_number", 0),
            bounding_box=annotation.bounding_box,
            area_type=annotation.area_type,
            confidence=annotation.confidence,
            image_path=image_path,
            annotated_by=annotation.metadata.get("annotated_by"),
            metadata=annotation.metadata,
        )

        # Save to database
        annotation_id = self.storage.save_annotation(annotation_record)

        logger.debug("Annotation saved with image: %s -> %s", annotation_id, image_path)
        return annotation_id

    def load_annotation(self, annotation_id: str) -> SensitiveArea | None:
        """Load annotation by ID and convert to SensitiveArea."""
        # Get all annotations to find the one we want
        # Note: This is a simple implementation - in practice you'd want direct lookup
        all_annotations = []

        # Get all videos and their annotations (this could be optimized)
        videos = self.storage.list_videos()
        for video in videos:
            video_annotations = self.storage.get_annotations_for_video(video.id)
            all_annotations.extend(video_annotations)

        # Find the specific annotation
        for annotation_record in all_annotations:
            if annotation_record.id == annotation_id:
                return self._convert_to_sensitive_area(annotation_record)

        return None

    def get_annotations_for_video(
        self, video_id: str, area_type: str | None = None
    ) -> list[SensitiveArea]:
        """Get all annotations for a video, optionally filtered by area type."""
        annotation_records = self.storage.get_annotations_for_video(video_id, area_type)
        return [
            self._convert_to_sensitive_area(record) for record in annotation_records
        ]

    def get_annotations_in_time_range(
        self, video_id: str, start_time: float, end_time: float
    ) -> list[SensitiveArea]:
        """Get annotations within a specific time range."""
        # This would need to be implemented in the storage service
        # For now, get all annotations and filter
        all_annotations = self.get_annotations_for_video(video_id)
        return [
            ann for ann in all_annotations if start_time <= ann.timestamp <= end_time
        ]

    def update_annotation(self, annotation: SensitiveArea) -> bool:
        """Update an existing annotation."""
        # Convert to record and save (will replace existing due to INSERT OR REPLACE)
        annotation_record = AnnotationRecord(
            id=annotation.id,
            video_id=annotation.metadata.get("video_id", ""),
            timestamp=annotation.timestamp,
            frame_number=annotation.metadata.get("frame_number", 0),
            bounding_box=annotation.bounding_box,
            area_type=annotation.area_type,
            confidence=annotation.confidence,
            image_path=Path(annotation.image_path) if annotation.image_path else None,
            annotated_by=annotation.metadata.get("annotated_by"),
            metadata=annotation.metadata,
        )

        try:
            self.storage.save_annotation(annotation_record)
            logger.info("Annotation updated: %s", annotation.id)
        except Exception:
            logger.exception("Failed to update annotation %s", annotation.id)
            return False
        return True

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation and associated files."""
        # First get the annotation to find associated files
        annotation = self.load_annotation(annotation_id)
        if not annotation:
            logger.warning("Annotation not found for deletion: %s", annotation_id)
            return False

        # Delete associated frame image if it exists
        if annotation.image_path and annotation.image_path.exists():
            success = self.file_manager.delete_file(annotation.image_path)
            if success:
                logger.debug("Deleted frame image: %s", annotation.image_path)
            else:
                logger.warning(
                    "Failed to delete frame image: %s", annotation.image_path
                )

        # Delete from database (would need to be implemented in storage service)
        # For now, we'll log the operation
        logger.info("Annotation marked for deletion: %s", annotation_id)

        # TODO: Implement actual database deletion in StorageService
        return True

    def get_annotation_statistics(self, video_id: str) -> dict:
        """Get statistics about annotations for a video."""
        annotations = self.get_annotations_for_video(video_id)

        stats = {
            "total_count": len(annotations),
            "by_area_type": {},
            "confidence_stats": {"mean": 0.0, "min": 1.0, "max": 0.0},
            "time_coverage": {
                "earliest": float("inf"),
                "latest": 0.0,
                "total_span": 0.0,
            },
        }

        if not annotations:
            return stats

        # Calculate area type distribution
        for annotation in annotations:
            area_type = annotation.area_type
            stats["by_area_type"][area_type] = (
                stats["by_area_type"].get(area_type, 0) + 1
            )

        # Calculate confidence statistics
        confidences = [ann.confidence for ann in annotations]
        stats["confidence_stats"] = {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
        }

        # Calculate time coverage
        timestamps = [ann.timestamp for ann in annotations]
        stats["time_coverage"] = {
            "earliest": min(timestamps),
            "latest": max(timestamps),
            "total_span": max(timestamps) - min(timestamps),
        }

        return stats

    def batch_import_annotations(
        self,
        annotations: list[SensitiveArea],
        frame_images: list[np.ndarray] | None = None,
    ) -> list[str]:
        """Import multiple annotations efficiently."""
        imported_ids = []

        for i, annotation in enumerate(annotations):
            frame_image = (
                frame_images[i] if frame_images and i < len(frame_images) else None
            )

            try:
                annotation_id = self.save_annotation(annotation, frame_image)
                imported_ids.append(annotation_id)
            except Exception:
                logger.exception("Failed to import annotation %s", annotation.id)
                continue

        logger.info(
            "Batch import completed: %s/%s annotations imported",
            len(imported_ids),
            len(annotations),
        )
        return imported_ids

    def export_annotations(self, video_id: str, export_format: str = "json") -> Path:
        """Export all annotations for a video to a file."""
        annotations = self.get_annotations_for_video(video_id)

        # Convert annotations to exportable format
        export_data = {
            "video_id": video_id,
            "export_timestamp": self.storage.get_current_timestamp(),
            "annotation_count": len(annotations),
            "annotations": [
                {
                    "id": ann.id,
                    "timestamp": ann.timestamp,
                    "bounding_box": {
                        "x": ann.bounding_box.x,
                        "y": ann.bounding_box.y,
                        "width": ann.bounding_box.width,
                        "height": ann.bounding_box.height,
                    },
                    "area_type": ann.area_type,
                    "confidence": ann.confidence,
                    "metadata": ann.metadata,
                }
                for ann in annotations
            ],
        }

        # Use file manager to store export
        export_path = self.file_manager.store_timeline_export(
            export_data, f"annotations_{video_id}", export_format
        )

        logger.info("Annotations exported: %s -> %s", video_id, export_path)
        return export_path

    def _convert_to_sensitive_area(self, record: AnnotationRecord) -> SensitiveArea:
        """Convert AnnotationRecord to SensitiveArea."""
        # Update metadata with database fields
        metadata = record.metadata.copy()
        metadata.update(
            {
                "video_id": record.video_id,
                "frame_number": record.frame_number,
                "annotated_by": record.annotated_by,
                "created_at": record.created_at.isoformat(),
            }
        )

        return SensitiveArea(
            id=record.id,
            timestamp=record.timestamp,
            bounding_box=record.bounding_box,
            area_type=record.area_type,
            confidence=record.confidence,
            image_path=record.image_path,
            metadata=metadata,
        )
