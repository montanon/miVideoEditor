"""Ensemble detector for combining multiple detection algorithms."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from mivideoeditor.core.models import BoundingBox, DetectionResult, SensitiveArea
from mivideoeditor.detection.base import BaseDetector, DetectionConfig, DetectionError

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble detection."""

    strategy: str = Field(default="voting", pattern="^(voting|cascade|stacking)$")
    confidence_weights: dict[str, float] = Field(default_factory=dict)
    min_consensus_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_confidence_boosting: bool = Field(default=True)
    cascade_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("confidence_weights")
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure all weights are positive."""
        if not all(weight > 0 for weight in v.values()):
            msg = "All confidence weights must be positive"
            raise ValueError(msg)
        return v

    class Config:
        """Configuration for ensemble detection."""

        frozen = True


class DetectionCluster(BaseModel):
    """Cluster of overlapping detections."""

    detections: list[tuple[BoundingBox, float, str]] = Field(..., min_items=1)
    center_bbox: BoundingBox
    combined_confidence: float = Field(..., ge=0.0, le=1.0)
    detector_consensus: dict[str, int] = Field(default_factory=dict)

    class Config:
        """Configuration for detection cluster."""

        arbitrary_types_allowed = True
        frozen = True

    @property
    def detection_count(self) -> int:
        """Get number of detections in cluster."""
        return len(self.detections)

    @property
    def detector_types(self) -> set[str]:
        """Get unique detector types in cluster."""
        return {detector_type for _, _, detector_type in self.detections}

    @property
    def average_confidence(self) -> float:
        """Get average confidence of detections in cluster."""
        if not self.detections:
            return 0.0
        return sum(conf for _, conf, _ in self.detections) / len(self.detections)


class EnsembleDetector(BaseDetector):
    """Combine multiple detectors using ensemble methods."""

    def __init__(
        self,
        detectors: list[BaseDetector],
        config: DetectionConfig,
        ensemble_config: EnsembleConfig | None = None,
    ) -> None:
        super().__init__(config)

        if not detectors:
            msg = "At least one detector must be provided"
            raise ValueError(msg)

        self.detectors = detectors
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.detector_weights = self._calculate_detector_weights()

        # Set ensemble as trained if all detectors are trained
        self.is_trained = all(detector.is_trained for detector in self.detectors)

    def _calculate_detector_weights(self) -> dict[str, float]:
        """Calculate weights for each detector based on performance or config."""
        weights = {}

        for detector in self.detectors:
            detector_name = detector.__class__.__name__

            # Use configured weight if available
            if detector_name in self.ensemble_config.confidence_weights:
                weights[detector_name] = self.ensemble_config.confidence_weights[
                    detector_name
                ]
            else:
                # Default weight based on detector performance stats
                stats = detector.get_detection_stats()
                avg_confidence = stats.get("average_confidence", 0.5)
                # Higher average confidence gets higher weight
                weights[detector_name] = max(0.1, avg_confidence)

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Run ensemble detection."""
        start_time = time.time()

        try:
            self._validate_frame(frame)

            if self.ensemble_config.strategy == "voting":
                return self._voting_ensemble(frame, timestamp, start_time)
            if self.ensemble_config.strategy == "cascade":
                return self._cascade_ensemble(frame, timestamp, start_time)
            if self.ensemble_config.strategy == "stacking":
                return self._stacking_ensemble(frame, timestamp, start_time)
            msg = f"Unknown ensemble strategy: {self.ensemble_config.strategy}"
            raise DetectionError(msg)

        except Exception as e:
            logger.exception("Ensemble detection failed")
            detection_time = time.time() - start_time
            error_result = DetectionResult(
                regions=[],
                confidences=[],
                detection_time=detection_time,
                detector_type=f"ensemble_{self.ensemble_config.strategy}",
                timestamp=timestamp,
                frame_metadata={"error": str(e)},
            )
            self._update_stats(error_result)
            return error_result

    def _voting_ensemble(
        self, frame: np.ndarray, timestamp: float, start_time: float
    ) -> DetectionResult:
        """Weighted voting ensemble."""
        # Get results from all detectors
        individual_results = []
        total_detection_time = 0.0

        for detector in self.detectors:
            try:
                result = detector.detect(frame, timestamp)
                individual_results.append(result)
                total_detection_time += result.detection_time
            except (DetectionError, RuntimeError, ValueError) as e:
                logger.warning("Detector %s failed: %s", detector.__class__.__name__, e)
                # Create empty result for failed detector
                empty_result = DetectionResult(
                    regions=[],
                    confidences=[],
                    detection_time=0.0,
                    detector_type=detector.__class__.__name__,
                    timestamp=timestamp,
                    frame_metadata={"error": str(e)},
                )
                individual_results.append(empty_result)

        if not individual_results:
            return self._create_empty_result(timestamp, time.time() - start_time)

        # Collect all detections with weights
        weighted_detections = []
        for result in individual_results:
            detector_type = result.detector_type
            weight = self.detector_weights.get(detector_type, 0.1)

            for bbox, conf in zip(result.regions, result.confidences, strict=False):
                weighted_detections.append((bbox, conf * weight, detector_type))

        if not weighted_detections:
            return self._create_empty_result(timestamp, time.time() - start_time)

        # Cluster nearby detections
        clusters = self._cluster_detections(weighted_detections)

        # Generate final detections from clusters
        final_regions = []
        final_confidences = []

        for cluster in clusters:
            # Filter clusters by consensus
            consensus_ratio = len(cluster.detector_types) / len(self.detectors)
            if consensus_ratio < self.ensemble_config.min_consensus_ratio:
                continue

            final_regions.append(cluster.center_bbox)
            final_confidences.append(min(cluster.combined_confidence, 1.0))

        # Limit number of detections
        if len(final_regions) > self.config.max_regions_per_frame:
            # Sort by confidence and take top N
            sorted_indices = sorted(
                range(len(final_confidences)),
                key=lambda i: final_confidences[i],
                reverse=True,
            )[: self.config.max_regions_per_frame]

            final_regions = [final_regions[i] for i in sorted_indices]
            final_confidences = [final_confidences[i] for i in sorted_indices]

        detection_time = time.time() - start_time
        result = DetectionResult(
            regions=final_regions,
            confidences=final_confidences,
            detection_time=detection_time,
            detector_type=f"ensemble_{self.ensemble_config.strategy}",
            timestamp=timestamp,
            frame_metadata={
                "individual_results": len(individual_results),
                "clusters_formed": len(clusters),
                "clusters_accepted": len(final_regions),
                "total_individual_detection_time": total_detection_time,
            },
        )

        self._update_stats(result)
        return result

    def _cascade_ensemble(
        self, frame: np.ndarray, timestamp: float, start_time: float
    ) -> DetectionResult:
        """Cascade ensemble: fast detector filters candidates for slow detector."""
        if len(self.detectors) < 2:
            # Fallback to single detector
            result = self.detectors[0].detect(frame, timestamp)
            self._update_stats(result)
            return result

        # Use first detector as fast filter
        fast_detector = self.detectors[0]
        slow_detectors = self.detectors[1:]

        # Get initial detections from fast detector
        fast_result = fast_detector.detect(frame, timestamp)

        # Filter by confidence threshold
        candidate_regions = []
        for bbox, conf in zip(
            fast_result.regions, fast_result.confidences, strict=False
        ):
            if conf >= self.ensemble_config.cascade_confidence_threshold:
                candidate_regions.append((bbox, conf))

        if not candidate_regions:
            # No candidates passed fast detector
            detection_time = time.time() - start_time
            result = DetectionResult(
                regions=[],
                confidences=[],
                detection_time=detection_time,
                detector_type="ensemble_cascade",
                timestamp=timestamp,
                frame_metadata={
                    "fast_detector_regions": len(fast_result.regions),
                    "candidates_passed": 0,
                },
            )
            self._update_stats(result)
            return result

        # Run slow detectors on candidate regions
        final_detections = []

        for bbox, conf in candidate_regions:
            # Extract region for detailed analysis
            region_image = self._extract_region(frame, bbox)

            if region_image.size == 0:
                continue

            # Get consensus from slow detectors
            slow_results = []
            for slow_detector in slow_detectors:
                try:
                    slow_result = slow_detector.detect(region_image, timestamp)
                    slow_results.append(slow_result)
                except (DetectionError, RuntimeError, ValueError, OSError) as e:
                    logger.warning(
                        "Slow detector %s failed: %s",
                        slow_detector.__class__.__name__,
                        e,
                    )

            # Combine results
            if slow_results:
                # Use highest confidence from slow detectors
                best_confidence = max(
                    max(result.confidences) if result.confidences else 0.0
                    for result in slow_results
                )

                # Boost confidence if multiple slow detectors agree
                agreement_count = sum(
                    1
                    for result in slow_results
                    if result.confidences
                    and max(result.confidences) > self.config.confidence_threshold
                )

                if agreement_count > 0:
                    confidence_boost = 1.0 + (agreement_count - 1) * 0.1
                    final_confidence = min(best_confidence * confidence_boost, 1.0)
                    final_detections.append((bbox, final_confidence))

        detection_time = time.time() - start_time
        result = DetectionResult(
            regions=[bbox for bbox, _ in final_detections],
            confidences=[conf for _, conf in final_detections],
            detection_time=detection_time,
            detector_type="ensemble_cascade",
            timestamp=timestamp,
            frame_metadata={
                "fast_detector_regions": len(fast_result.regions),
                "candidates_passed": len(candidate_regions),
                "final_detections": len(final_detections),
            },
        )

        self._update_stats(result)
        return result

    def _stacking_ensemble(
        self, frame: np.ndarray, timestamp: float, start_time: float
    ) -> DetectionResult:
        """Stacking ensemble: use one detector's output as input to another."""
        # This is a simplified stacking implementation
        # In practice, you might train a meta-model to combine detector outputs

        all_results = []
        combined_detections = []

        # Collect results from all detectors
        for detector in self.detectors:
            try:
                result = detector.detect(frame, timestamp)
                all_results.append(result)

                # Add detections with detector type info
                for bbox, conf in zip(result.regions, result.confidences, strict=False):
                    combined_detections.append(
                        (bbox, conf, detector.__class__.__name__)
                    )
            except (DetectionError, RuntimeError, ValueError, OSError) as e:
                logger.warning(
                    "Detector %s failed in stacking: %s", detector.__class__.__name__, e
                )

        # Simple stacking: weight by detector performance and combine
        final_detections = []

        for bbox, conf, detector_type in combined_detections:
            weight = self.detector_weights.get(detector_type, 0.1)

            # Meta-learner logic (simplified)
            # In practice, this would be a trained model
            meta_confidence = self._calculate_meta_confidence(
                bbox, detector_type, all_results
            )

            final_confidence = conf * weight * meta_confidence

            if final_confidence > self.config.confidence_threshold:
                final_detections.append((bbox, final_confidence))

        # Remove duplicates and sort
        final_detections = self._remove_duplicate_detections(final_detections)
        final_detections.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        if len(final_detections) > self.config.max_regions_per_frame:
            final_detections = final_detections[: self.config.max_regions_per_frame]

        detection_time = time.time() - start_time
        result = DetectionResult(
            regions=[bbox for bbox, _ in final_detections],
            confidences=[conf for _, conf in final_detections],
            detection_time=detection_time,
            detector_type="ensemble_stacking",
            timestamp=timestamp,
            frame_metadata={
                "individual_detectors": len(all_results),
                "combined_detections": len(combined_detections),
                "final_detections": len(final_detections),
            },
        )

        self._update_stats(result)
        return result

    def _cluster_detections(
        self, weighted_detections: list[tuple[BoundingBox, float, str]]
    ) -> list[DetectionCluster]:
        """Cluster nearby detections."""
        if not weighted_detections:
            return []

        clusters = []
        used_detections = set()

        for i, (bbox1, conf1, detector1) in enumerate(weighted_detections):
            if i in used_detections:
                continue

            # Start new cluster
            cluster_detections = [(bbox1, conf1, detector1)]
            used_detections.add(i)

            # Find overlapping detections
            for j, (bbox2, conf2, detector2) in enumerate(
                weighted_detections[i + 1 :], i + 1
            ):
                if j in used_detections:
                    continue

                iou = bbox1.iou(bbox2)
                if iou > self.ensemble_config.iou_threshold:
                    cluster_detections.append((bbox2, conf2, detector2))
                    used_detections.add(j)

            # Create cluster
            if cluster_detections:
                cluster = self._create_cluster(cluster_detections)
                clusters.append(cluster)

        return clusters

    def _create_cluster(
        self, detections: list[tuple[BoundingBox, float, str]]
    ) -> DetectionCluster:
        """Create detection cluster from list of detections."""
        # Calculate weighted average bounding box
        total_weight = sum(conf for _, conf, _ in detections)

        if total_weight == 0:
            # Fallback to simple average
            center_bbox = self._average_bounding_boxes(
                [(bbox, 1.0) for bbox, _, _ in detections]
            )
            combined_confidence = 0.0
        else:
            weighted_boxes = [(bbox, conf) for bbox, conf, _ in detections]
            center_bbox = self._average_bounding_boxes(weighted_boxes)

            # Combined confidence with ensemble boosting
            base_confidence = total_weight / len(detections)

            if self.ensemble_config.enable_confidence_boosting:
                # Boost confidence based on number of agreeing detectors
                detector_types = {detector_type for _, _, detector_type in detections}
                boost_factor = 1.0 + (len(detector_types) - 1) * 0.1
                combined_confidence = min(base_confidence * boost_factor, 1.0)
            else:
                combined_confidence = base_confidence

        # Count detector consensus
        detector_consensus = {}
        for _, _, detector_type in detections:
            detector_consensus[detector_type] = (
                detector_consensus.get(detector_type, 0) + 1
            )

        return DetectionCluster(
            detections=detections,
            center_bbox=center_bbox,
            combined_confidence=combined_confidence,
            detector_consensus=detector_consensus,
        )

    def _average_bounding_boxes(
        self, weighted_boxes: list[tuple[BoundingBox, float]]
    ) -> BoundingBox:
        """Calculate weighted average of bounding boxes."""
        if not weighted_boxes:
            return BoundingBox(0, 0, 0, 0)

        total_weight = sum(weight for _, weight in weighted_boxes)
        if total_weight == 0:
            # Simple average
            boxes = [bbox for bbox, _ in weighted_boxes]
            avg_x = sum(box.x for box in boxes) / len(boxes)
            avg_y = sum(box.y for box in boxes) / len(boxes)
            avg_w = sum(box.width for box in boxes) / len(boxes)
            avg_h = sum(box.height for box in boxes) / len(boxes)
        else:
            # Weighted average
            avg_x = sum(box.x * weight for box, weight in weighted_boxes) / total_weight
            avg_y = sum(box.y * weight for box, weight in weighted_boxes) / total_weight
            avg_w = (
                sum(box.width * weight for box, weight in weighted_boxes) / total_weight
            )
            avg_h = (
                sum(box.height * weight for box, weight in weighted_boxes)
                / total_weight
            )

        return BoundingBox(
            x=int(avg_x),
            y=int(avg_y),
            width=int(avg_w),
            height=int(avg_h),
        )

    def _extract_region(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract image region from frame."""
        try:
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(frame.shape[1], bbox.x + bbox.width)
            y2 = min(frame.shape[0], bbox.y + bbox.height)

            if x2 <= x1 or y2 <= y1:
                return np.array([])  # Empty array

            return frame[y1:y2, x1:x2]
        except (IndexError, ValueError, OSError) as e:
            logger.warning("Failed to extract region %s: %s", bbox, e)
            return np.array([])

    def _calculate_meta_confidence(
        self,
        bbox: BoundingBox,
        detector_type: str,
        all_results: list[DetectionResult],
    ) -> float:
        """Calculate meta-learner confidence (simplified implementation)."""
        # Simplified meta-learning: boost confidence if multiple detectors agree
        agreement_score = 0.0

        for result in all_results:
            if result.detector_type == detector_type:
                continue

            for other_bbox, other_conf in zip(
                result.regions, result.confidences, strict=False
            ):
                iou = bbox.iou(other_bbox)
                if iou > 0.5:  # Good overlap
                    agreement_score += iou * other_conf

        # Normalize and combine with base confidence
        normalized_agreement = min(agreement_score, 1.0)
        return 0.7 + 0.3 * normalized_agreement  # 0.7 to 1.0 range

    def _remove_duplicate_detections(
        self, detections: list[tuple[BoundingBox, float]]
    ) -> list[tuple[BoundingBox, float]]:
        """Remove duplicate/overlapping detections."""
        if not detections:
            return []

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)

        final_detections = []

        for bbox, conf in sorted_detections:
            is_duplicate = False

            for existing_bbox, _ in final_detections:
                if bbox.iou(existing_bbox) > self.ensemble_config.iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_detections.append((bbox, conf))

        return final_detections

    def _create_empty_result(
        self, timestamp: float, detection_time: float
    ) -> DetectionResult:
        """Create empty detection result."""
        result = DetectionResult(
            regions=[],
            confidences=[],
            detection_time=detection_time,
            detector_type=f"ensemble_{self.ensemble_config.strategy}",
            timestamp=timestamp,
        )
        self._update_stats(result)
        return result

    def train(self, annotations: list[SensitiveArea]) -> None:
        """Train all component detectors."""
        logger.info("Training ensemble with %d annotations", len(annotations))

        training_errors = []

        for detector in self.detectors:
            try:
                detector.train(annotations)
                logger.info("Successfully trained %s", detector.__class__.__name__)
            except Exception as e:
                error_msg = f"Failed to train {detector.__class__.__name__}: {e}"
                logger.exception(error_msg)
                training_errors.append(error_msg)

        # Update weights after training
        self.detector_weights = self._calculate_detector_weights()

        # Set ensemble as trained if at least one detector trained successfully
        self.is_trained = any(detector.is_trained for detector in self.detectors)

        if training_errors:
            if not self.is_trained:
                # All detectors failed
                msg = f"All detectors failed to train: {training_errors}"
                raise DetectionError(msg)
            # Some detectors failed, but ensemble can still work
            logger.warning("Some detectors failed to train: %s", training_errors)

        logger.info(
            "Ensemble training completed. Trained detectors: %d/%d",
            sum(1 for d in self.detectors if d.is_trained),
            len(self.detectors),
        )

    def get_detector_info(self) -> dict[str, Any]:
        """Get information about component detectors."""
        detector_info = {}

        for detector in self.detectors:
            detector_name = detector.__class__.__name__
            stats = detector.get_detection_stats()

            detector_info[detector_name] = {
                "is_trained": detector.is_trained,
                "weight": self.detector_weights.get(detector_name, 0.0),
                "stats": stats,
            }

        return {
            "ensemble_strategy": self.ensemble_config.strategy,
            "total_detectors": len(self.detectors),
            "trained_detectors": sum(1 for d in self.detectors if d.is_trained),
            "detectors": detector_info,
        }
