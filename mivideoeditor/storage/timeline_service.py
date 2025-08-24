"""Specialized storage service for timeline operations."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mivideoeditor.core.models import BlurRegion, BoundingBox, Timeline
from mivideoeditor.storage.file_manager import FileManager
from mivideoeditor.storage.models import TimelineRecord
from mivideoeditor.storage.service import StorageService

logger = logging.getLogger(__name__)


class TimelineService:
    """High-level service for timeline storage and versioning operations."""

    def __init__(self, storage_service: StorageService, file_manager: FileManager):
        self.storage = storage_service
        self.file_manager = file_manager

    def save_timeline(self, timeline: Timeline, created_by: str | None = None) -> str:
        """Save timeline with automatic versioning."""
        # Check if timeline already exists
        existing_timeline = self.get_latest_timeline(timeline.video_path)

        version = 1
        if existing_timeline:
            version = existing_timeline.version + 1
            logger.debug("Creating new timeline version: %s", version)

        # Serialize timeline data
        timeline_data = self._serialize_timeline(timeline)

        # Get video ID from path (in practice, you'd look this up)
        video_id = self._get_video_id_from_path(timeline.video_path)

        # Create timeline record
        timeline_record = TimelineRecord(
            id=timeline.id,
            video_id=video_id,
            name=timeline.metadata.get("name", f"Timeline v{version}"),
            version=version,
            timeline_data=timeline_data,
            status="draft",
            created_by=created_by,
            metadata=timeline.metadata,
        )

        # Save to database
        timeline_id = self._save_timeline_record(timeline_record)

        # Export timeline data to file for backup/reference
        export_path = self.file_manager.store_timeline_export(
            timeline_data, f"{timeline_id}_v{version}", "json"
        )

        logger.info("Timeline saved: %s v%s -> %s", timeline_id, version, export_path)
        return timeline_id

    def load_timeline(
        self, timeline_id: str, version: int | None = None
    ) -> Timeline | None:
        """Load timeline by ID, optionally specific version."""
        timeline_record = self._get_timeline_record(timeline_id, version)

        if not timeline_record:
            return None

        return self._deserialize_timeline(timeline_record)

    def get_latest_timeline(self, video_path: Path) -> TimelineRecord | None:
        """Get the latest version of timeline for a video."""
        video_id = self._get_video_id_from_path(video_path)

        # This would need to be implemented in storage service
        # For now, return None as placeholder
        return None

    def get_timeline_versions(self, timeline_id: str) -> list[dict[str, Any]]:
        """Get all versions of a timeline with metadata."""
        # This would query the database for all versions of a timeline
        # For now, return empty list as placeholder
        return []

    def approve_timeline(self, timeline_id: str, approved_by: str) -> bool:
        """Mark timeline as approved for processing."""
        try:
            # This would update the database status
            # For now, we'll simulate the approval
            logger.info("Timeline approved: %s by %s", timeline_id, approved_by)
        except Exception as e:
            logger.exception("Failed to approve timeline %s: %s", timeline_id, e)
            return False
        return True

    def archive_timeline(self, timeline_id: str, version: int | None = None) -> bool:
        """Archive a timeline version."""
        try:
            # This would update the database status to 'archived'
            logger.info("Timeline archived: %s v%s", timeline_id, version or "latest")
        except Exception as e:
            logger.exception("Failed to archive timeline %s: %s", timeline_id, e)
            return False
        return True

    def duplicate_timeline(self, source_timeline_id: str, new_name: str) -> str | None:
        """Create a duplicate of an existing timeline."""
        source_timeline = self.load_timeline(source_timeline_id)

        if not source_timeline:
            logger.error("Source timeline not found: %s", source_timeline_id)
            return None

        # Create new timeline with modified metadata
        new_timeline = Timeline(
            video_path=source_timeline.video_path,
            video_duration=source_timeline.video_duration,
            frame_rate=source_timeline.frame_rate,
            blur_regions=source_timeline.blur_regions.copy(),
            version=1,  # New timeline starts at version 1
            metadata={
                **source_timeline.metadata,
                "name": new_name,
                "duplicated_from": source_timeline_id,
                "duplicated_at": datetime.now(UTC).isoformat(),
            },
        )

        return self.save_timeline(new_timeline)

    def merge_timelines(self, timeline_ids: list[str], merged_name: str) -> str | None:
        """Merge multiple timelines into a single timeline."""
        if len(timeline_ids) < 2:
            logger.error("At least 2 timelines required for merging")
            return None

        timelines = []
        for timeline_id in timeline_ids:
            timeline = self.load_timeline(timeline_id)
            if timeline:
                timelines.append(timeline)
            else:
                logger.warning("Timeline not found for merging: %s", timeline_id)

        if not timelines:
            logger.error("No valid timelines found for merging")
            return None

        # Use the first timeline as base
        base_timeline = timelines[0]

        # Merge blur regions from all timelines
        merged_regions = base_timeline.blur_regions.copy()
        for timeline in timelines[1:]:
            merged_regions.extend(timeline.blur_regions)

        # Create merged timeline
        merged_timeline = Timeline(
            video_path=base_timeline.video_path,
            video_duration=base_timeline.video_duration,
            frame_rate=base_timeline.frame_rate,
            blur_regions=merged_regions,
            version=1,
            metadata={
                "name": merged_name,
                "merged_from": timeline_ids,
                "merged_at": datetime.now(UTC).isoformat(),
                "total_blur_regions": len(merged_regions),
            },
        )

        merged_id = self.save_timeline(merged_timeline)
        logger.info("Timelines merged: %s -> %s", timeline_ids, merged_id)
        return merged_id

    def export_timeline(self, timeline_id: str, export_format: str = "json") -> Path:
        """Export timeline to external format."""
        timeline = self.load_timeline(timeline_id)

        if not timeline:
            msg = f"Timeline not found: {timeline_id!r}"
            raise ValueError(msg)

        export_data = self._serialize_timeline(timeline)
        export_data["export_metadata"] = {
            "timeline_id": timeline_id,
            "exported_at": datetime.now(UTC).isoformat(),
            "export_format": export_format,
        }

        export_path = self.file_manager.store_timeline_export(
            export_data, f"timeline_export_{timeline_id}", export_format
        )

        logger.info("Timeline exported: %s -> %s", timeline_id, export_path)
        return export_path

    def import_timeline(self, import_path: Path, video_id: str) -> str | None:
        """Import timeline from external file."""
        try:
            with import_path.open("r") as f:
                timeline_data = json.load(f)

            # Create timeline record from imported data
            timeline_record = TimelineRecord(
                video_id=video_id,
                name=timeline_data.get("name", "Imported Timeline"),
                timeline_data=timeline_data,
                status="draft",
                metadata={
                    "imported_from": str(import_path),
                    "imported_at": datetime.now(UTC).isoformat(),
                },
            )

            timeline_id = self._save_timeline_record(timeline_record)
            logger.info("Timeline imported: %s -> %s", import_path, timeline_id)

        except Exception:
            logger.exception("Failed to import timeline from %s", import_path)
            return None
        return timeline_id

    def get_timeline_statistics(self, timeline_id: str) -> dict[str, Any]:
        """Get detailed statistics about a timeline."""
        timeline = self.load_timeline(timeline_id)

        if not timeline:
            return {"error": "Timeline not found"}

        stats = {
            "timeline_id": timeline_id,
            "blur_regions_count": len(timeline.blur_regions),
            "total_duration": timeline.video_duration,
            "blur_coverage": {"total_blur_time": 0.0, "coverage_percentage": 0.0},
            "blur_types": {},
            "time_distribution": {"earliest_blur": float("inf"), "latest_blur": 0.0},
        }

        if timeline.blur_regions:
            # Calculate blur coverage
            total_blur_time = 0.0
            for region in timeline.blur_regions:
                blur_duration = region.end_time - region.start_time
                total_blur_time += blur_duration

                # Track blur types
                blur_type = region.blur_type
                stats["blur_types"][blur_type] = (
                    stats["blur_types"].get(blur_type, 0) + 1
                )

                # Track time distribution
                stats["time_distribution"]["earliest_blur"] = min(
                    stats["time_distribution"]["earliest_blur"], region.start_time
                )
                stats["time_distribution"]["latest_blur"] = max(
                    stats["time_distribution"]["latest_blur"], region.end_time
                )

            stats["blur_coverage"] = {
                "total_blur_time": total_blur_time,
                "coverage_percentage": (total_blur_time / timeline.video_duration)
                * 100,
            }

        return stats

    def _serialize_timeline(self, timeline: Timeline) -> dict[str, Any]:
        """Convert Timeline object to serializable dictionary."""
        return {
            "video_path": str(timeline.video_path),
            "video_duration": timeline.video_duration,
            "frame_rate": timeline.frame_rate,
            "blur_regions": [
                {
                    "id": region.id,
                    "start_time": region.start_time,
                    "end_time": region.end_time,
                    "bounding_box": {
                        "x": region.bounding_box.x,
                        "y": region.bounding_box.y,
                        "width": region.bounding_box.width,
                        "height": region.bounding_box.height,
                    },
                    "blur_type": region.blur_type,
                    "blur_strength": region.blur_strength,
                    "interpolation": region.interpolation,
                    "confidence": region.confidence,
                    "needs_review": region.needs_review,
                    "metadata": region.metadata,
                }
                for region in timeline.blur_regions
            ],
            "version": timeline.version,
            "created_at": timeline.created_at.isoformat(),
            "metadata": timeline.metadata,
        }

    def _deserialize_timeline(self, record: TimelineRecord) -> Timeline:
        """Convert TimelineRecord to Timeline object."""
        timeline_data = record.timeline_data

        # Deserialize blur regions
        blur_regions = []
        for region_data in timeline_data.get("blur_regions", []):
            bbox_data = region_data["bounding_box"]
            bounding_box = BoundingBox(
                x=bbox_data["x"],
                y=bbox_data["y"],
                width=bbox_data["width"],
                height=bbox_data["height"],
            )

            blur_region = BlurRegion(
                id=region_data["id"],
                start_time=region_data["start_time"],
                end_time=region_data["end_time"],
                bounding_box=bounding_box,
                blur_type=region_data["blur_type"],
                blur_strength=region_data["blur_strength"],
                interpolation=region_data["interpolation"],
                confidence=region_data["confidence"],
                needs_review=region_data["needs_review"],
                metadata=region_data["metadata"],
            )
            blur_regions.append(blur_region)

        return Timeline(
            id=record.id,
            video_path=Path(timeline_data["video_path"]),
            video_duration=timeline_data["video_duration"],
            frame_rate=timeline_data["frame_rate"],
            blur_regions=blur_regions,
            version=record.version,
            created_at=record.created_at,
            metadata=record.metadata,
        )

    def _save_timeline_record(self, record: TimelineRecord) -> str:
        """Save TimelineRecord to database."""
        # This would use the storage service to save the record
        # For now, we'll simulate the save operation
        logger.debug("Saving timeline record: %s", record.id)
        return record.id

    def _get_timeline_record(
        self, timeline_id: str, version: int | None = None
    ) -> TimelineRecord | None:
        """Get timeline record from database."""
        # This would query the storage service
        # For now, return None as placeholder
        return None

    def _get_video_id_from_path(self, video_path: Path) -> str:
        """Get video ID from video path (lookup in database)."""
        # This would query the video database to find the ID
        # For now, use a simple hash of the path as placeholder
        return hashlib.md5(str(video_path).encode()).hexdigest()[:8]
