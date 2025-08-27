"""Detection module - Algorithms for identifying sensitive areas in video frames."""

from mivideoeditor.detection import constants
from mivideoeditor.detection.base import (
    BaseDetector,
    DetectionConfig,
    DetectionError,
    DetectionTimeoutError,
    TrainingError,
    create_detection_result_empty,
    validate_detection_config,
)
from mivideoeditor.detection.cnn import (
    CNNDetector,
    YOLODetector,
)
from mivideoeditor.detection.ensemble import (
    DetectionCluster,
    EnsembleConfig,
    EnsembleDetector,
)
from mivideoeditor.detection.motion_tracker import (
    MotionTracker,
    Track,
    TrackedDetection,
    TrackingStats,
)
from mivideoeditor.detection.template import (
    ColorProfile,
    Detection,
    TemplateDetector,
)
from mivideoeditor.detection.training import (
    AnnotationStatistics,
    DetectorTrainer,
    TemplateTrainer,
    TrainingConfig,
    TrainingDataProcessor,
    TrainingResult,
)

__all__ = [
    # Base detection framework
    "BaseDetector",
    "DetectionConfig",
    "DetectionError",
    "TrainingError",
    "DetectionTimeoutError",
    "create_detection_result_empty",
    "validate_detection_config",
    # Template detection
    "TemplateDetector",
    "ColorProfile",
    "Detection",
    # Motion tracking
    "MotionTracker",
    "TrackedDetection",
    "Track",
    "TrackingStats",
    # Ensemble detection
    "EnsembleDetector",
    "EnsembleConfig",
    "DetectionCluster",
    # Training system
    "DetectorTrainer",
    "TrainingConfig",
    "TrainingResult",
    "AnnotationStatistics",
    "TrainingDataProcessor",
    "TemplateTrainer",
    # Deep learning detectors
    "CNNDetector",
    "YOLODetector",
    "constants",
]
