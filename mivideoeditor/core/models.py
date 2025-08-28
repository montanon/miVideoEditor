"""Core data models for video privacy editor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from mivideoeditor.core.constants import (
    BLUR_FILTER_TYPES,
    INTERPOLATION_MODES,
    SUPPORTED_AREA_TYPES,
)


class BoundingBox(BaseModel):
    """Represents a rectangular region in pixel coordinates."""

    x: int = Field(..., ge=0, description="Left coordinate (0-based)")
    y: int = Field(..., ge=0, description="Top coordinate (0-based)")
    width: int = Field(..., gt=0, description="Box width (must be > 0)")
    height: int = Field(..., gt=0, description="Box height (must be > 0)")

    model_config = {
        "frozen": True,  # Make immutable
        "str_strip_whitespace": True,
    }

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
            detection for detection in self.detections if detection[1] >= threshold
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
            detection for detection in self.detections if detection[2] == area_type
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


class BlurRegion(BaseModel):
    """Represents a temporal region to be blurred in the final video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4)",
    )
    start_time: float = Field(..., ge=0, description="Start timestamp (seconds)")
    end_time: float = Field(..., ge=0, description="End timestamp (seconds)")
    bounding_box: BoundingBox = Field(..., description="Region to blur")
    blur_type: str = Field(
        default="gaussian",
        description="Filter: gaussian, pixelate, noise, composite",
    )
    blur_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Strength multiplier [0.0, 2.0]",
    )
    interpolation: str = Field(
        default="linear",
        description="Motion interpolation: linear, smooth, none",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence",
    )
    needs_review: bool = Field(
        default=False,
        description="Flag for manual review",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Ensure end_time is after start_time."""
        if self.end_time <= self.start_time:
            msg = (
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )
            raise ValueError(msg)
        return self

    @field_validator("blur_type")
    @classmethod
    def validate_blur_type(cls, v: str) -> str:
        """Validate blur type against supported types."""
        if v not in BLUR_FILTER_TYPES:
            supported = list(BLUR_FILTER_TYPES.keys())
            msg = f"Unsupported blur type: {v}. Must be one of {supported}"
            raise ValueError(msg)
        return v

    @field_validator("interpolation")
    @classmethod
    def validate_interpolation(cls, v: str) -> str:
        """Validate interpolation mode."""
        if v not in INTERPOLATION_MODES:
            modes = INTERPOLATION_MODES
            msg = f"Unsupported interpolation: {v}. Must be one of {modes}"
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

    @property
    def duration(self) -> float:
        """Calculate the duration in seconds."""
        return self.end_time - self.start_time

    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        return self.bounding_box.area

    def overlaps_time(self, timestamp: float) -> bool:
        """Check if timestamp is within this blur region."""
        return self.start_time <= timestamp <= self.end_time

    def overlaps_time_range(self, start: float, end: float) -> bool:
        """Check if this region overlaps with a time range."""
        return not (self.end_time <= start or end <= self.start_time)

    def get_region_at_time(self, timestamp: float) -> BoundingBox:
        """Get region position at specific timestamp (for motion interpolation)."""
        if not self.overlaps_time(timestamp):
            msg = (
                f"Timestamp {timestamp} not within region "
                f"{self.start_time}-{self.end_time}"
            )
            raise ValueError(msg)

        # For now, return the same bounding box (no motion)
        # Future enhancement: implement actual motion interpolation
        return self.bounding_box

    def split_at_time(self, timestamp: float) -> tuple[BlurRegion, BlurRegion]:
        """Split region at specific timestamp into two regions."""
        if not (self.start_time < timestamp < self.end_time):
            msg = (
                f"Cannot split at {timestamp}, must be within "
                f"{self.start_time}-{self.end_time}"
            )
            raise ValueError(msg)

        first_region = self.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "end_time": timestamp,
            }
        )

        second_region = self.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "start_time": timestamp,
            }
        )

        return first_region, second_region

    def merge_with(self, other: BlurRegion) -> BlurRegion:
        """Merge with another overlapping blur region."""
        if not self.can_merge_with(other):
            msg = "Cannot merge non-overlapping or incompatible regions"
            raise ValueError(msg)

        # Take union of bounding boxes
        merged_bbox = self.bounding_box.union(other.bounding_box)

        return BlurRegion(
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
            bounding_box=merged_bbox,
            blur_type=self.blur_type,  # Keep first region's blur type
            blur_strength=max(self.blur_strength, other.blur_strength),
            confidence=min(self.confidence, other.confidence),
            needs_review=self.needs_review or other.needs_review,
            metadata={"merged_from": [self.id, other.id]},
        )

    def can_merge_with(self, other: BlurRegion) -> bool:
        """Check if this region can be merged with another."""
        # Must have same blur type and overlapping time
        return (
            self.blur_type == other.blur_type
            and self.overlaps_time_range(other.start_time, other.end_time)
            and self.bounding_box.overlaps(other.bounding_box)
        )

    def with_updated_bbox(self, new_bbox: BoundingBox) -> BlurRegion:
        """Create a copy with updated bounding box."""
        return self.model_copy(update={"bounding_box": new_bbox})

    def with_updated_strength(self, new_strength: float) -> BlurRegion:
        """Create a copy with updated blur strength."""
        return self.model_copy(update={"blur_strength": new_strength})

    @classmethod
    def from_sensitive_area(
        cls,
        area: SensitiveArea,
        duration: float = 1.0,
        blur_type: str = "gaussian",
    ) -> BlurRegion:
        """Create a BlurRegion from a SensitiveArea."""
        return cls(
            start_time=area.timestamp,
            end_time=area.timestamp + duration,
            bounding_box=area.bounding_box,
            blur_type=blur_type,
            confidence=area.confidence,
            needs_review=area.needs_review,
            metadata={"source_area_id": area.id, "area_type": area.area_type},
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"BlurRegion({self.blur_type} {self.start_time:.1f}-{self.end_time:.1f}s, "
            f"strength={self.blur_strength:.1f})"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"BlurRegion(id={self.id[:8]}..., time={self.start_time}-{self.end_time}, "
            f"bbox={self.bounding_box}, type={self.blur_type}, "
            f"strength={self.blur_strength})"
        )


class Timeline(BaseModel):
    """Complete timeline of blur operations for a video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4)",
    )
    video_path: Path = Field(..., description="Source video file")
    video_duration: float = Field(..., gt=0, description="Total video length (seconds)")
    frame_rate: float = Field(
        default=30.0,
        gt=0,
        description="Video frame rate",
    )
    blur_regions: list[BlurRegion] = Field(
        default_factory=list,
        description="All blur operations",
    )
    version: str = Field(default="1.0", description="Version for compatibility")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional timeline metadata",
    )

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, v: Path) -> Path:
        """Validate video path."""
        # Convert to Path if string, don't check existence as file might not exist yet
        return Path(v)

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

    @model_validator(mode="after")
    def validate_blur_regions(self) -> Self:
        """Validate blur regions don't exceed video duration."""
        for region in self.blur_regions:
            if region.end_time > self.video_duration:
                msg = (
                    f"Blur region extends beyond video duration: "
                    f"{region.end_time} > {self.video_duration}"
                )
                raise ValueError(msg)
        return self

    def get_active_regions(self, timestamp: float) -> list[BlurRegion]:
        """Get all blur regions active at given timestamp."""
        return [
            region for region in self.blur_regions if region.overlaps_time(timestamp)
        ]

    def get_regions_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[BlurRegion]:
        """Get all regions that overlap with time range."""
        return [
            region
            for region in self.blur_regions
            if region.overlaps_time_range(start_time, end_time)
        ]

    def add_region(self, region: BlurRegion) -> None:
        """Add a new blur region."""
        if region.end_time > self.video_duration:
            msg = (
                f"Cannot add region extending beyond video duration: "
                f"{region.end_time} > {self.video_duration}"
            )
            raise ValueError(msg)
        self.blur_regions.append(region)

    def remove_region(self, region_id: str) -> bool:
        """Remove a blur region by ID."""
        for i, region in enumerate(self.blur_regions):
            if region.id == region_id:
                del self.blur_regions[i]
                return True
        return False

    def total_blur_duration(self) -> float:
        """Sum of all blur durations (may count overlaps multiple times)."""
        return sum(region.duration for region in self.blur_regions)

    def blur_coverage_percentage(self) -> float:
        """Percentage of video that has blur (considering overlaps)."""
        if self.video_duration == 0:
            return 0.0

        # Create time segments and mark which are covered
        segment_size = 0.1  # 100ms segments
        total_segments = int(self.video_duration / segment_size) + 1
        covered_segments = set()

        for region in self.blur_regions:
            start_segment = int(region.start_time / segment_size)
            end_segment = int(region.end_time / segment_size) + 1
            for seg in range(start_segment, end_segment):
                if seg < total_segments:
                    covered_segments.add(seg)

        return (len(covered_segments) / total_segments) * 100.0

    def optimize(self) -> Timeline:
        """Create an optimized version by merging overlapping compatible regions."""
        if not self.blur_regions:
            return self

        # Group regions by blur type for potential merging
        optimized_regions: list[BlurRegion] = []
        remaining_regions = self.blur_regions.copy()

        while remaining_regions:
            current = remaining_regions.pop(0)

            # Try to merge with other regions
            merged_any = True
            while merged_any:
                merged_any = False
                for i, other in enumerate(remaining_regions):
                    if current.can_merge_with(other):
                        current = current.merge_with(other)
                        remaining_regions.pop(i)
                        merged_any = True
                        break

            optimized_regions.append(current)

        return self.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "blur_regions": optimized_regions,
                "metadata": {
                    **self.metadata,
                    "optimized_from": self.id,
                    "original_region_count": len(self.blur_regions),
                },
            }
        )

    def validate(self) -> list[str]:
        """Validate timeline and return list of issues."""
        issues = []

        # Check for overlapping regions that can't be merged
        for i, region1 in enumerate(self.blur_regions):
            for region2 in self.blur_regions[i + 1 :]:
                if (
                    region1.overlaps_time_range(region2.start_time, region2.end_time)
                    and region1.bounding_box.overlaps(region2.bounding_box)
                    and not region1.can_merge_with(region2)
                ):
                    msg = (
                        f"Incompatible overlapping regions: "
                        f"{region1.id[:8]} and {region2.id[:8]}"
                    )
                    issues.append(msg)

        # Check for regions needing review
        review_needed = [r for r in self.blur_regions if r.needs_review]
        if review_needed:
            issues.append(f"{len(review_needed)} regions need manual review")

        # Check for low confidence regions
        confidence_threshold = 0.7
        low_confidence = [
            r for r in self.blur_regions if r.confidence < confidence_threshold
        ]
        if low_confidence:
            issues.append(
                f"{len(low_confidence)} regions have low confidence "
                f"(<{confidence_threshold})"
            )

        return issues

    @property
    def region_count(self) -> int:
        """Get the number of blur regions."""
        return len(self.blur_regions)

    @property
    def has_regions(self) -> bool:
        """Check if timeline has any regions."""
        return len(self.blur_regions) > 0

    @property
    def needs_review(self) -> bool:
        """Check if any regions need review."""
        return any(region.needs_review for region in self.blur_regions)

    @classmethod
    def from_detection_results(
        cls,
        *,
        video_path: Path,
        video_duration: float,
        detection_results: list[DetectionResult],
        frame_rate: float = 30.0,
        default_duration: float = 1.0,
    ) -> Timeline:
        """Create timeline from detection results."""
        blur_regions = []

        for result in detection_results:
            sensitive_areas = result.to_sensitive_areas()
            for area in sensitive_areas:
                region = BlurRegion.from_sensitive_area(
                    area,
                    duration=default_duration,
                    blur_type="gaussian",  # Default blur type
                )
                blur_regions.append(region)

        return cls(
            video_path=video_path,
            video_duration=video_duration,
            frame_rate=frame_rate,
            blur_regions=blur_regions,
            metadata={
                "generated_from": "detection_results",
                "detection_count": sum(r.detection_count for r in detection_results),
            },
        )

    def __str__(self) -> str:
        """Return string representation."""
        coverage = self.blur_coverage_percentage()
        return (
            f"Timeline({self.region_count} regions, "
            f"{self.total_blur_duration():.1f}s blur, "
            f"{coverage:.1f}% coverage)"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"Timeline(id={self.id[:8]}..., video={self.video_path.name}, "
            f"duration={self.video_duration}s, regions={self.region_count})"
        )


class TimeRangeAnnotation(BaseModel):
    """Represents a sensitive area that persists across a time range."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4)",
    )
    start_time: float = Field(..., ge=0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0, description="End timestamp in seconds")
    bounding_box: BoundingBox = Field(..., description="Region coordinates")
    area_type: str = Field(..., description="Type: chatgpt, atuin, terminal, custom")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0]",
    )
    sample_frame_paths: list[Path] = Field(
        default_factory=list,
        description="Paths to sample frame images from the range",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and context",
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
            msg = f"area_type must be one of {list(SUPPORTED_AREA_TYPES.keys())}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        if self.end_time <= self.start_time:
            msg = f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            raise ValueError(msg)
        return self

    @property
    def duration(self) -> float:
        """Get the duration of the time range in seconds."""
        return self.end_time - self.start_time

    @property
    def center_timestamp(self) -> float:
        """Get the center timestamp of the range."""
        return (self.start_time + self.end_time) / 2

    def contains_timestamp(self, timestamp: float) -> bool:
        """Check if a timestamp falls within this range."""
        return self.start_time <= timestamp <= self.end_time

    def overlaps_with(self, other: TimeRangeAnnotation) -> bool:
        """Check if this range overlaps with another time range."""
        return not (
            self.end_time <= other.start_time or self.start_time >= other.end_time
        )

    def get_overlap_duration(self, other: TimeRangeAnnotation) -> float:
        """Get the duration of overlap with another time range."""
        if not self.overlaps_with(other):
            return 0.0

        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        return overlap_end - overlap_start

    def split_at_timestamp(
        self, timestamp: float
    ) -> tuple[TimeRangeAnnotation, TimeRangeAnnotation]:
        """Split this annotation at a specific timestamp."""
        if not self.contains_timestamp(timestamp):
            msg = f"Timestamp {timestamp} is not within range [{self.start_time}, {self.end_time}]"
            raise ValueError(msg)

        # Create first part
        first_part = TimeRangeAnnotation(
            start_time=self.start_time,
            end_time=timestamp,
            bounding_box=self.bounding_box,
            area_type=self.area_type,
            confidence=self.confidence,
            metadata={**self.metadata, "split_from": self.id},
        )

        # Create second part
        second_part = TimeRangeAnnotation(
            start_time=timestamp,
            end_time=self.end_time,
            bounding_box=self.bounding_box,
            area_type=self.area_type,
            confidence=self.confidence,
            metadata={**self.metadata, "split_from": self.id},
        )

        return first_part, second_part

    def to_sensitive_areas(self, sample_interval: float = 1.0) -> list[SensitiveArea]:
        """Convert to individual SensitiveArea annotations at regular intervals."""
        areas = []
        current_time = self.start_time

        while current_time <= self.end_time:
            # Use actual timestamp, but don't exceed end_time
            timestamp = min(current_time, self.end_time)

            area = SensitiveArea(
                timestamp=timestamp,
                bounding_box=self.bounding_box,
                area_type=self.area_type,
                confidence=self.confidence,
                metadata={
                    **self.metadata,
                    "source_range_id": self.id,
                    "range_start": self.start_time,
                    "range_end": self.end_time,
                },
            )
            areas.append(area)

            current_time += sample_interval

            # Ensure we include the end timestamp
            if current_time > self.end_time and timestamp < self.end_time:
                end_area = SensitiveArea(
                    timestamp=self.end_time,
                    bounding_box=self.bounding_box,
                    area_type=self.area_type,
                    confidence=self.confidence,
                    metadata={
                        **self.metadata,
                        "source_range_id": self.id,
                        "range_start": self.start_time,
                        "range_end": self.end_time,
                    },
                )
                areas.append(end_area)
                break

        return areas

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"TimeRange({self.area_type}, "
            f"{self.start_time:.1f}s-{self.end_time:.1f}s, "
            f"duration={self.duration:.1f}s)"
        )

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"TimeRangeAnnotation(id={self.id[:8]}..., "
            f"range=[{self.start_time}, {self.end_time}], "
            f"type={self.area_type}, bbox={self.bounding_box})"
        )
