"""Training system for detection algorithms."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from mivideoeditor.core.models import SensitiveArea
from mivideoeditor.detection.base import BaseDetector, TrainingError

# Training quality thresholds
EXCELLENT_ACCURACY_THRESHOLD = 0.9
EXCELLENT_TEMPLATE_COUNT = 5
GOOD_ACCURACY_THRESHOLD = 0.8
GOOD_TEMPLATE_COUNT = 3
FAIR_ACCURACY_THRESHOLD = 0.7
MIN_TEMPLATE_RECOMMENDATION = 3
REVIEW_ACCURACY_THRESHOLD = 0.8
SLOW_TRAINING_THRESHOLD = 300  # 5 minutes

# Template generation constants
RGB_CHANNELS = 3

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Configuration for detector training."""

    min_annotations_per_type: int = Field(default=5, ge=1)
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    template_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_templates_per_type: int = Field(default=10, ge=1)
    enable_data_augmentation: bool = Field(default=False)
    augmentation_factor: int = Field(default=3, ge=1, le=10)

    class Config:
        """Configuration for training config."""

        frozen = True

    @field_validator("validation_split")
    def validate_split(cls, v: float) -> float:
        """Ensure validation split is reasonable."""
        if v >= 0.5:
            msg = (
                "Validation split should be less than 0.5 "
                "to maintain enough training data"
            )
            raise ValueError(msg)
        return v


class AnnotationStatistics(BaseModel):
    """Statistics about annotation dataset."""

    total_annotations: int = Field(..., ge=0)
    annotations_by_type: dict[str, int] = Field(default_factory=dict)
    area_statistics: dict[str, dict[str, float]] = Field(default_factory=dict)
    confidence_distribution: dict[str, float] = Field(default_factory=dict)
    temporal_distribution: dict[str, int] = Field(default_factory=dict)

    class Config:
        """Configuration for annotation statistics."""

        frozen = True

    @property
    def unique_area_types(self) -> set[str]:
        """Get unique area types in dataset."""
        return set(self.annotations_by_type.keys())

    @property
    def average_confidence(self) -> float:
        """Get average confidence across all annotations."""
        if not self.confidence_distribution:
            return 0.0

        total_conf = sum(
            conf * count for conf, count in self.confidence_distribution.items()
        )
        total_count = sum(self.confidence_distribution.values())

        return total_conf / total_count if total_count > 0 else 0.0


class TrainingResult(BaseModel):
    """Result of training process."""

    success: bool
    trained_area_types: list[str] = Field(default_factory=list)
    templates_generated: dict[str, int] = Field(default_factory=dict)
    training_statistics: AnnotationStatistics
    validation_accuracy: dict[str, float] = Field(default_factory=dict)
    training_time_seconds: float = Field(..., ge=0.0)
    error_message: str | None = None

    class Config:
        """Configuration for training result."""

        frozen = True

    @property
    def total_templates(self) -> int:
        """Get total number of templates generated."""
        return sum(self.templates_generated.values())


class TrainingDataProcessor(BaseModel):
    """Process and validate training data."""

    config: TrainingConfig

    class Config:
        """Configuration for training data processor."""

        arbitrary_types_allowed = True

    def process_annotations(
        self, annotations: list[SensitiveArea]
    ) -> tuple[AnnotationStatistics, dict[str, list[SensitiveArea]]]:
        """Process and validate annotations for training.

        Returns:
            Tuple of (statistics, grouped_annotations)

        """
        if not annotations:
            msg = "No annotations provided for training"
            raise TrainingError(msg)

        # Calculate statistics
        stats = self._calculate_annotation_statistics(annotations)

        # Validate minimum annotations per type
        insufficient_types = []
        for area_type, count in stats.annotations_by_type.items():
            if count < self.config.min_annotations_per_type:
                insufficient_types.append((area_type, count))

        if insufficient_types:
            msg = (
                f"Insufficient annotations for training. Need at least "
                f"{self.config.min_annotations_per_type} annotations per type. "
                f"Found: {insufficient_types}"
            )
            raise TrainingError(msg)

        # Group annotations by type
        grouped_annotations: dict[str, list[SensitiveArea]] = {}
        for annotation in annotations:
            area_type = annotation.area_type
            if area_type not in grouped_annotations:
                grouped_annotations[area_type] = []
            grouped_annotations[area_type].append(annotation)

        logger.info(
            "Processed %d annotations across %d area types",
            len(annotations),
            len(grouped_annotations),
        )

        return stats, grouped_annotations

    def _calculate_annotation_statistics(
        self, annotations: list[SensitiveArea]
    ) -> AnnotationStatistics:
        """Calculate comprehensive statistics about annotations."""
        # Basic counts
        annotations_by_type: dict[str, int] = {}
        area_stats: dict[str, dict[str, float]] = {}
        confidence_dist: dict[str, float] = {}
        temporal_dist: dict[str, int] = {}

        for annotation in annotations:
            area_type = annotation.area_type

            # Count by type
            annotations_by_type[area_type] = annotations_by_type.get(area_type, 0) + 1

            # Area statistics
            if area_type not in area_stats:
                area_stats[area_type] = {
                    "total_area": 0,
                    "avg_width": 0,
                    "avg_height": 0,
                    "count": 0,
                }

            bbox = annotation.bounding_box
            area_stats[area_type]["total_area"] += bbox.area
            area_stats[area_type]["avg_width"] += bbox.width
            area_stats[area_type]["avg_height"] += bbox.height
            area_stats[area_type]["count"] += 1

            # Confidence distribution (binned)
            conf_bin = f"{annotation.confidence:.1f}"
            confidence_dist[conf_bin] = confidence_dist.get(conf_bin, 0) + 1

            # Temporal distribution (by hour of timestamp)
            if annotation.timestamp > 0:
                hour_bin = int(annotation.timestamp // 3600) % 24
                temporal_dist[f"hour_{hour_bin:02d}"] = (
                    temporal_dist.get(f"hour_{hour_bin:02d}", 0) + 1
                )

        # Calculate averages for area statistics
        for area_type, stats in area_stats.items():
            count = stats["count"]
            if count > 0:
                stats["avg_area"] = stats["total_area"] / count
                stats["avg_width"] = stats["avg_width"] / count
                stats["avg_height"] = stats["avg_height"] / count

        return AnnotationStatistics(
            total_annotations=len(annotations),
            annotations_by_type=annotations_by_type,
            area_statistics=area_stats,
            confidence_distribution=confidence_dist,
            temporal_distribution=temporal_dist,
        )

    def split_training_validation(
        self, grouped_annotations: dict[str, list[SensitiveArea]]
    ) -> tuple[dict[str, list[SensitiveArea]], dict[str, list[SensitiveArea]]]:
        """Split annotations into training and validation sets."""
        training_data: dict[str, list[SensitiveArea]] = {}
        validation_data: dict[str, list[SensitiveArea]] = {}

        for area_type, area_annotations in grouped_annotations.items():
            # Shuffle annotations (deterministically)
            sorted_annotations = sorted(
                area_annotations, key=lambda x: x.id
            )  # Deterministic ordering

            # Calculate split point
            total_count = len(sorted_annotations)
            validation_count = max(1, int(total_count * self.config.validation_split))
            training_count = total_count - validation_count

            # Split
            training_data[area_type] = sorted_annotations[:training_count]
            validation_data[area_type] = sorted_annotations[training_count:]

            logger.debug(
                "Split %s: %d training, %d validation",
                area_type,
                training_count,
                validation_count,
            )

        return training_data, validation_data


class TemplateTrainer(BaseModel):
    """Generate and optimize templates from annotations."""

    config: TrainingConfig

    class Config:
        """Configuration for template trainer."""

        arbitrary_types_allowed = True

    def generate_templates(
        self, training_data: dict[str, list[SensitiveArea]]
    ) -> dict[str, list[np.ndarray]]:
        """Generate templates from training annotations."""
        templates = {}

        for area_type, training_annotations in training_data.items():
            logger.info(
                "Generating templates for %s from %d annotations",
                area_type,
                len(training_annotations),
            )

            # Placeholder template generation
            # In reality, this would:
            # 1. Extract image regions from frames based on bounding boxes
            # 2. Normalize and preprocess the regions
            # 3. Cluster similar regions
            # 4. Select representative templates

            type_templates = []

            for i, annotation in enumerate(
                training_annotations[: self.config.max_templates_per_type]
            ):
                # Create placeholder template based on bounding box size
                width = annotation.bounding_box.width
                height = annotation.bounding_box.height

                # Normalize to common size range
                template_width = min(max(width, 50), 200)
                template_height = min(max(height, 30), 150)

                # Create synthetic template (extracted from frame in practice)
                template = self._create_synthetic_template(
                    template_width, template_height, area_type, i
                )
                type_templates.append(template)

            templates[area_type] = type_templates
            logger.info("Generated %d templates for %s", len(type_templates), area_type)

        return templates

    def _create_synthetic_template(
        self, width: int, height: int, area_type: str, index: int
    ) -> np.ndarray:
        """Create a synthetic template for testing (placeholder)."""
        # Create a synthetic template with patterns based on area type
        template = np.zeros((height, width, 3), dtype=np.uint8)

        if area_type == "chatgpt":
            # Dark background with some structure
            template[:, :] = [26, 26, 26]  # Dark gray
            # Add some vertical structure (sidebar-like)
            if width > 50:
                template[:, :20] = [45, 45, 45]  # Lighter sidebar
        elif area_type == "atuin":
            # Terminal-like background
            template[:, :] = [12, 12, 12]  # Very dark
            # Add some horizontal lines (terminal text)
            for i in range(5, height, 15):
                if i < height - 2:
                    template[i : i + 2, :] = [180, 180, 180]  # Light text lines
        else:
            # Generic template
            template[:, :] = [100, 100, 100]  # Medium gray

        # Add some variation based on index
        noise_level = (index * 10) % 30
        noise = np.random.randint(
            -noise_level, noise_level + 1, template.shape, dtype=np.int16
        )
        return np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def optimize_templates(
        self, templates: dict[str, list[np.ndarray]]
    ) -> dict[str, list[np.ndarray]]:
        """Optimize templates for better matching performance."""
        optimized_templates: dict[str, list[np.ndarray]] = {}

        for area_type, type_templates in templates.items():
            logger.info(
                "Optimizing %d templates for %s", len(type_templates), area_type
            )

            optimized_list = []

            for template in type_templates:
                # Apply optimization techniques
                optimized = self._apply_template_optimizations(template)
                optimized_list.append(optimized)

            # Remove very similar templates
            deduplicated = self._remove_similar_templates(optimized_list)

            optimized_templates[area_type] = deduplicated
            logger.info(
                "Optimized templates for %s: %d -> %d",
                area_type,
                len(type_templates),
                len(deduplicated),
            )

        return optimized_templates

    def _apply_template_optimizations(self, template: np.ndarray) -> np.ndarray:
        """Apply optimization techniques to a single template."""
        # Convert to grayscale for template matching
        if len(template.shape) == RGB_CHANNELS:
            # Simple RGB to grayscale conversion
            grayscale = np.dot(template[..., :3], [0.299, 0.587, 0.114])
            template = grayscale.astype(np.uint8)

        # Apply slight Gaussian blur to reduce noise
        # Note: In a real implementation, you'd use cv2.GaussianBlur
        # Here we just return the template as-is for simplicity

        return template

    def _remove_similar_templates(
        self, templates: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Remove templates that are too similar to each other."""
        if len(templates) <= 1:
            return templates

        unique_templates = []

        for template in templates:
            is_unique = True

            for existing in unique_templates:
                # Simple similarity check (in reality, use proper template matching)
                if template.shape == existing.shape:
                    # Calculate simple MSE similarity
                    mse = np.mean(
                        (template.astype(float) - existing.astype(float)) ** 2
                    )
                    normalized_mse = mse / (255.0 * 255.0)  # Normalize to [0, 1]

                    similarity = 1.0 - normalized_mse

                    if similarity > self.config.template_similarity_threshold:
                        is_unique = False
                        break

            if is_unique:
                unique_templates.append(template)

        return unique_templates


class DetectorTrainer(BaseModel):
    """Main training coordinator for detection algorithms."""

    config: TrainingConfig
    data_processor: TrainingDataProcessor
    template_trainer: TemplateTrainer

    class Config:
        """Configuration for detector trainer."""

        arbitrary_types_allowed = True

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize detector trainer."""
        config = config or TrainingConfig()
        super().__init__(
            config=config,
            data_processor=TrainingDataProcessor(config=config),
            template_trainer=TemplateTrainer(config=config),
        )

    def train_detector(
        self,
        detector: BaseDetector,
        annotations: list[SensitiveArea],
        save_path: Path | None = None,
    ) -> TrainingResult:
        """Train a detector with annotations."""
        start_time = time.time()

        try:
            logger.info(
                "Starting detector training with %d annotations", len(annotations)
            )

            # Process and validate annotations
            stats, grouped_annotations = self.data_processor.process_annotations(
                annotations
            )

            # Split into training and validation
            training_data, validation_data = (
                self.data_processor.split_training_validation(grouped_annotations)
            )

            # Generate templates
            templates = self.template_trainer.generate_templates(training_data)
            optimized_templates = self.template_trainer.optimize_templates(templates)

            # Train the detector
            detector.train(
                annotations
            )  # Use original annotations for detector training

            # Calculate validation accuracy (simplified)
            validation_accuracy = self._calculate_validation_accuracy(
                detector, validation_data
            )

            # Save model if requested
            if save_path:
                detector.save_model(save_path)
                logger.info("Model saved to %s", save_path)

            training_time = time.time() - start_time

            result = TrainingResult(
                success=True,
                trained_area_types=list(grouped_annotations.keys()),
                templates_generated={
                    area_type: len(templates)
                    for area_type, templates in optimized_templates.items()
                },
                training_statistics=stats,
                validation_accuracy=validation_accuracy,
                training_time_seconds=training_time,
            )

            logger.info(
                "Training completed successfully in %.2f seconds", training_time
            )
            return result

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Training failed: {e}"
            logger.exception(error_msg)

            return TrainingResult(
                success=False,
                training_statistics=AnnotationStatistics(
                    total_annotations=len(annotations)
                ),
                training_time_seconds=training_time,
                error_message=error_msg,
            )

    def _calculate_validation_accuracy(
        self, _detector: BaseDetector, validation_data: dict[str, list[SensitiveArea]]
    ) -> dict[str, float]:
        """Calculate validation accuracy for each area type."""
        # This is a simplified placeholder implementation

        accuracy = {}

        for area_type, annotation_list in validation_data.items():
            # Placeholder: assume 85% accuracy with some variation
            base_accuracy = 0.85
            variation = (
                len(annotation_list) * 0.01
            )  # More data = slightly better accuracy

            calculated_accuracy = min(base_accuracy + variation, 0.95)
            accuracy[area_type] = calculated_accuracy

            logger.debug(
                "Validation accuracy for %s: %.2f", area_type, calculated_accuracy
            )

        return accuracy

    def evaluate_training_quality(self, result: TrainingResult) -> dict[str, Any]:
        """Evaluate the quality of training results."""
        evaluation = {
            "overall_quality": "unknown",
            "recommendations": [],
            "metrics": {},
        }

        if not result.success:
            evaluation["overall_quality"] = "failed"
            evaluation["recommendations"].append("Fix training errors and retry")
            return evaluation

        # Evaluate based on various factors
        total_templates = result.total_templates
        avg_accuracy = (
            sum(result.validation_accuracy.values()) / len(result.validation_accuracy)
            if result.validation_accuracy
            else 0.0
        )

        evaluation["metrics"] = {
            "total_templates": total_templates,
            "average_accuracy": avg_accuracy,
            "area_types_trained": len(result.trained_area_types),
            "training_time": result.training_time_seconds,
        }

        # Quality assessment
        if (
            avg_accuracy >= EXCELLENT_ACCURACY_THRESHOLD
            and total_templates >= EXCELLENT_TEMPLATE_COUNT
        ):
            evaluation["overall_quality"] = "excellent"
        elif (
            avg_accuracy >= GOOD_ACCURACY_THRESHOLD
            and total_templates >= GOOD_TEMPLATE_COUNT
        ):
            evaluation["overall_quality"] = "good"
        elif avg_accuracy >= FAIR_ACCURACY_THRESHOLD:
            evaluation["overall_quality"] = "fair"
        else:
            evaluation["overall_quality"] = "poor"

        # Recommendations
        if total_templates < MIN_TEMPLATE_RECOMMENDATION:
            evaluation["recommendations"].append(
                "Consider collecting more training data"
            )

        if avg_accuracy < REVIEW_ACCURACY_THRESHOLD:
            evaluation["recommendations"].append(
                "Review annotation quality and consistency"
            )

        if result.training_time_seconds > SLOW_TRAINING_THRESHOLD:
            evaluation["recommendations"].append(
                "Training time is high, consider optimization"
            )

        return evaluation
