"""Core data models for video privacy editor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from mivideoeditor.core.constants import SUPPORTED_AREA_TYPES


class BoundingBox(BaseModel):
    """Represents a rectangular region in pixel coordinates."""

    x: int = Field(..., ge=0, description="Left coordinate (0-based)")
    y: int = Field(..., ge=0, description="Top coordinate (0-based)")
    width: int = Field(..., gt=0, description="Box width (must be > 0)")
    height: int = Field(..., gt=0, description="Box height (must be > 0)")

    class Config:
        """Pydantic configuration."""

        frozen = True  # Make immutable
        str_strip_whitespace = True

    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Get the center coordinates of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def x2(self) -> int:
        """Get the right coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Get the bottom coordinate."""
        return self.y + self.height

    @property
    def corners(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Get all four corner coordinates."""
        return (
            (self.x, self.y),
            (self.x2, self.y),
            (self.x2, self.y2),
            (self.x, self.y2),
        )

    def contains(self, point: tuple[int, int]) -> bool:
        """Check if a point is inside this bounding box."""
        px, py = point
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def overlaps(self, other: BoundingBox) -> bool:
        """Check if this bounding box overlaps with another."""
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        """Calculate the intersection with another bounding box."""
        if not self.overlaps(other):
            return None

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def union(self, other: BoundingBox) -> BoundingBox:
        """Calculate the union with another bounding box."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)

        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another bounding box."""
        intersection_box = self.intersection(other)
        if intersection_box is None:
            return 0.0

        intersection_area = intersection_box.area
        union_area = self.area + other.area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def expand(self, pixels: int) -> BoundingBox:
        """Expand the bounding box by the specified number of pixels."""
        return BoundingBox(
            x=max(0, self.x - pixels),
            y=max(0, self.y - pixels),
            width=self.width + 2 * pixels,
            height=self.height + 2 * pixels,
        )

    def scale(self, scale_x: float, scale_y: float | None = None) -> BoundingBox:
        """Scale the bounding box by the specified factors."""
        if scale_y is None:
            scale_y = scale_x

        new_width = int(self.width * scale_x)
        new_height = int(self.height * scale_y)

        # Adjust position to keep center fixed
        center_x, center_y = self.center
        new_x = center_x - new_width // 2
        new_y = center_y - new_height // 2

        return BoundingBox(
            x=max(0, new_x),
            y=max(0, new_y),
            width=max(1, new_width),
            height=max(1, new_height),
        )

    def clip(self, frame_width: int, frame_height: int) -> BoundingBox:
        """Clip the bounding box to fit within frame dimensions."""
        x = max(0, min(self.x, frame_width - 1))
        y = max(0, min(self.y, frame_height - 1))
        x2 = max(x + 1, min(self.x2, frame_width))
        y2 = max(y + 1, min(self.y2, frame_height))

        return BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)

    def to_normalized(self, frame_width: int, frame_height: int) -> dict[str, float]:
        """Convert to normalized coordinates [0, 1]."""
        return {
            "x": self.x / frame_width,
            "y": self.y / frame_height,
            "width": self.width / frame_width,
            "height": self.height / frame_height,
        }

    @classmethod
    def from_normalized(
        cls,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        """Create from normalized coordinates [0, 1]."""
        return cls(
            x=int(x * frame_width),
            y=int(y * frame_height),
            width=max(1, int(width * frame_width)),
            height=max(1, int(height * frame_height)),
        )

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
        """Create from corner coordinates."""
        return cls(
            x=min(x1, x2),
            y=min(y1, y2),
            width=abs(x2 - x1),
            height=abs(y2 - y1),
        )

    @model_validator(mode="after")
    def validate_frame_bounds(self) -> Self:
        """Additional validation for bounding box."""
        # Maximum reasonable frame size (8K resolution)
        max_dimension = 7680
        if self.x > max_dimension or self.y > max_dimension:
            msg = f"Coordinates exceed maximum dimension ({max_dimension})"
            raise ValueError(msg)
        if self.x + self.width > max_dimension or self.y + self.height > max_dimension:
            msg = f"Bounding box extends beyond maximum dimension ({max_dimension})"
            raise ValueError(msg)
        return self

    def __str__(self) -> str:
        """Return string representation."""
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, "
            f"height={self.height}, area={self.area})"
        )


class ValidationResult(BaseModel):
    """Result of validation operations with errors and warnings."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: list[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    context: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional context about validation",
    )

    def add_error(self, error: str) -> None:
        """Add an error message and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message without affecting validity."""
        self.warnings.append(warning)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge with another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            context={**self.context, **other.context},
        )

    @classmethod
    def success(cls, warnings: list[str] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, warnings=warnings or [])

    @classmethod
    def failure(cls, errors: list[str] | str) -> ValidationResult:
        """Create a failed validation result."""
        if isinstance(errors, str):
            errors = [errors]
        return cls(is_valid=False, errors=errors)

    @field_validator("errors", "warnings")
    @classmethod
    def validate_messages(cls, v: list[str]) -> list[str]:
        """Ensure all messages are non-empty strings."""
        return [msg for msg in v if msg and isinstance(msg, str)]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def message_count(self) -> int:
        """Get total number of errors and warnings."""
        return len(self.errors) + len(self.warnings)

    def raise_if_invalid(self, exception_class: type[Exception] = ValueError) -> None:
        """Raise an exception if validation failed."""
        if not self.is_valid:
            error_msg = f"Validation failed: {'; '.join(self.errors)}"
            raise exception_class(error_msg)

    def __str__(self) -> str:
        """Return string representation."""
        status = "valid" if self.is_valid else "invalid"
        return (
            f"ValidationResult({status}, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings)"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"ValidationResult(is_valid={self.is_valid}, "
            f"errors={self.errors!r}, warnings={self.warnings!r})"
        )


class SensitiveArea(BaseModel):
    """Represents an annotated sensitive region with metadata."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4)",
    )
    timestamp: float = Field(..., ge=0, description="Video timestamp in seconds")
    bounding_box: BoundingBox = Field(..., description="Region coordinates")
    area_type: str = Field(..., description="Type: chatgpt, atuin, terminal, custom")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0]",
    )
    image_path: Path | None = Field(
        default=None,
        description="Path to extracted frame image",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Validate area type against supported types."""
        if v not in SUPPORTED_AREA_TYPES:
            supported = list(SUPPORTED_AREA_TYPES.keys())
            msg = f"Unsupported area type: {v}. Must be one of {supported}"
            raise ValueError(msg)
        return v

    @field_validator("id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Validate UUID format."""
        try:
            uuid.UUID(v)
        except ValueError as e:
            msg = f"Invalid UUID format: {v}"
            raise ValueError(msg) from e
        return v

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Path | None) -> Path | None:
        """Validate image path exists if provided."""
        if v is not None and not v.exists():
            # Just convert to Path, don't check existence as file might not exist yet
            return Path(v)
        return v

    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        return self.bounding_box.area

    @property
    def is_manual(self) -> bool:
        """Check if this is a manual annotation (confidence = 1.0)."""
        return self.confidence == 1.0

    @property
    def needs_review(self) -> bool:
        """Check if this area needs manual review."""
        review_threshold = 0.8
        return self.confidence < review_threshold

    def to_detection_format(self) -> dict[str, Any]:
        """Convert to detection result format."""
        return {
            "timestamp": self.timestamp,
            "bounding_box": self.bounding_box.model_dump(),
            "confidence": self.confidence,
            "area_type": self.area_type,
            "metadata": self.metadata,
        }

    def with_updated_confidence(self, new_confidence: float) -> SensitiveArea:
        """Create a copy with updated confidence."""
        return self.model_copy(update={"confidence": new_confidence})

    def with_updated_bbox(self, new_bbox: BoundingBox) -> SensitiveArea:
        """Create a copy with updated bounding box."""
        return self.model_copy(update={"bounding_box": new_bbox})

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"SensitiveArea({self.area_type} at {self.timestamp:.2f}s, "
            f"confidence={self.confidence:.2f})"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"SensitiveArea(id={self.id[:8]}..., timestamp={self.timestamp}, "
            f"bbox={self.bounding_box}, type={self.area_type}, "
            f"confidence={self.confidence})"
        )


class DetectionResult(BaseModel):
    """Container for detection algorithm results from a single frame."""

    detections: list[tuple[BoundingBox, float, str]] = Field(
        default_factory=list,
        description="List of (region, confidence, area_type) tuples",
    )
    detection_time: float = Field(..., ge=0, description="Processing time in seconds")
    detector_type: str = Field(..., description="Which detector produced this")
    timestamp: float = Field(default=0.0, ge=0, description="Frame timestamp")
    frame_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional frame information",
    )

    @field_validator("detections")
    @classmethod
    def validate_detections(
        cls,
        v: list[tuple[BoundingBox, float, str]],
    ) -> list[tuple[BoundingBox, float, str]]:
        """Validate detection tuples."""
        for i, detection in enumerate(v):
            expected_length = 3
            if len(detection) != expected_length:
                msg = (
                    f"Detection at index {i} must be "
                    f"(BoundingBox, confidence, area_type)"
                )
                raise ValueError(msg)
            _, confidence, area_type = detection
            if not 0.0 <= confidence <= 1.0:
                msg = f"Confidence at index {i} out of range: {confidence}"
                raise ValueError(msg)
            if area_type not in SUPPORTED_AREA_TYPES:
                msg = f"Invalid area_type at index {i}: {area_type}"
                raise ValueError(msg)
        return v

    @property
    def regions(self) -> list[BoundingBox]:
        """Get all detected regions."""
        return [region for region, _, _ in self.detections]

    @property
    def confidences(self) -> list[float]:
        """Get all confidence scores."""
        return [conf for _, conf, _ in self.detections]

    @property
    def area_types(self) -> list[str]:
        """Get all area types."""
        return [area_type for _, _, area_type in self.detections]

    @property
    def best_detection(self) -> tuple[BoundingBox, float, str] | None:
        """Get the highest confidence detection."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda x: x[1])

    @property
    def detection_count(self) -> int:
        """Get the number of detections."""
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        """Check if there are any detections."""
        return len(self.detections) > 0

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all detections."""
        if not self.detections:
            return 0.0
        confidences = self.confidences
        return sum(confidences) / len(confidences)

    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        """Filter detections by confidence threshold."""
        filtered = [
            detection
            for detection in self.detections
            if detection[1] >= threshold
        ]

        return DetectionResult(
            detections=filtered,
            detection_time=self.detection_time,
            detector_type=self.detector_type,
            timestamp=self.timestamp,
            frame_metadata=self.frame_metadata,
        )

    def filter_by_area_type(self, area_type: str) -> DetectionResult:
        """Filter detections by area type."""
        filtered = [
            detection
            for detection in self.detections
            if detection[2] == area_type
        ]

        return DetectionResult(
            detections=filtered,
            detection_time=self.detection_time,
            detector_type=self.detector_type,
            timestamp=self.timestamp,
            frame_metadata=self.frame_metadata,
        )

    def merge_with(self, other: DetectionResult) -> DetectionResult:
        """Merge with another detection result."""
        return DetectionResult(
            detections=self.detections + other.detections,
            detection_time=self.detection_time + other.detection_time,
            detector_type=f"{self.detector_type}+{other.detector_type}",
            timestamp=self.timestamp,
            frame_metadata={**self.frame_metadata, **other.frame_metadata},
        )

    def to_sensitive_areas(self) -> list[SensitiveArea]:
        """Convert detections to SensitiveArea objects."""
        areas = []
        for region, confidence, area_type in self.detections:
            area = SensitiveArea(
                timestamp=self.timestamp,
                bounding_box=region,
                area_type=area_type,
                confidence=confidence,
                metadata={"detector": self.detector_type},
            )
            areas.append(area)
        return areas

    def add_detection(
        self,
        region: BoundingBox,
        confidence: float,
        area_type: str,
    ) -> None:
        """Add a new detection."""
        self.detections.append((region, confidence, area_type))

    @classmethod
    def empty(
        cls,
        timestamp: float = 0.0,
        detector_type: str = "unknown",
    ) -> DetectionResult:
        """Create an empty detection result."""
        return cls(
            detections=[],
            detection_time=0.0,
            detector_type=detector_type,
            timestamp=timestamp,
        )

    def __str__(self) -> str:
        """Return string representation."""
        type_counts = {}
        for _, _, area_type in self.detections:
            type_counts[area_type] = type_counts.get(area_type, 0) + 1
        types_str = ", ".join(f"{k}:{v}" for k, v in type_counts.items())
        return (
            f"DetectionResult({self.detection_count} detections [{types_str}], "
            f"avg_conf={self.average_confidence:.2f}, time={self.detection_time:.3f}s)"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"DetectionResult(detections={self.detection_count}, "
            f"detector={self.detector_type}, timestamp={self.timestamp}, "
            f"time={self.detection_time:.3f}s)"
        )
