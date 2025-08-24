"""Core module - Fundamental data structures and models."""

from mivideoeditor.core.constants import (
    BLUR_FILTER_TYPES,
    INTERPOLATION_MODES,
    SUPPORTED_AREA_TYPES,
)
from mivideoeditor.core.models import (
    BlurRegion,
    BoundingBox,
    DetectionResult,
    SensitiveArea,
    Timeline,
    ValidationResult,
)

__all__ = [
    # Models
    "BlurRegion",
    "BoundingBox",
    "DetectionResult",
    "SensitiveArea",
    "Timeline",
    "ValidationResult",
    # Constants
    "BLUR_FILTER_TYPES",
    "INTERPOLATION_MODES",
    "SUPPORTED_AREA_TYPES",
]
