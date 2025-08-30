"""Storage models using Pydantic for validation and serialization."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from mivideoeditor.core.constants import SUPPORTED_AREA_TYPES
from mivideoeditor.core.models import BoundingBox

TO_REVIEW_CONFIDENCE_THRESHOLD = 0.8


class VideoRecord(BaseModel):
    """Video metadata record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique video identifier"
    )
    filename: str = Field(..., min_length=1, description="Original filename")
    filepath: Path = Field(..., description="Path to video file")
    duration: float = Field(..., gt=0, description="Video duration in seconds")
    frame_rate: float = Field(..., gt=0, description="Video frame rate")
    width: int = Field(..., gt=0, description="Video width in pixels")
    height: int = Field(..., gt=0, description="Video height in pixels")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    codec: str | None = Field(None, description="Video codec")
    checksum: str | None = Field(None, description="File checksum for integrity")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, v: Path) -> Path:
        """Validate filepath is absolute."""
        return Path(v).resolve()

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str | None) -> str | None:
        """Validate checksum format if provided."""
        if v is not None and len(v) not in [32, 40, 64]:  # MD5, SHA1, SHA256
            msg = "Checksum must be 32, 40, or 64 characters long"
            raise ValueError(msg)
        return v

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height

    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return (self.width * self.height) / 1_000_000


class AnnotationRecord(BaseModel):
    """Annotation record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique annotation identifier",
    )
    video_id: str = Field(..., min_length=1, description="Associated video ID")
    timestamp: float = Field(..., ge=0, description="Timestamp in video (seconds)")
    frame_number: int = Field(..., ge=0, description="Frame number in video")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    area_type: str = Field(..., description="Type of sensitive area")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    image_path: Path | None = Field(None, description="Path to extracted frame image")
    annotated_by: str | None = Field(None, description="User who created annotation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Validate area type against supported types."""
        if v not in SUPPORTED_AREA_TYPES:
            msg = (
                f"Unsupported area type: {v}. Must be one of "
                f"{list(SUPPORTED_AREA_TYPES.keys())}"
            )
            raise ValueError(msg)
        return v

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Path | None) -> Path | None:
        """Convert to Path and validate if provided."""
        return Path(v) if v is not None else None

    @property
    def is_manual(self) -> bool:
        """Check if this is a manual annotation."""
        return self.confidence == 1.0

    @property
    def needs_review(self) -> bool:
        """Check if annotation needs review based on confidence."""
        return self.confidence < TO_REVIEW_CONFIDENCE_THRESHOLD


class DetectionRecord(BaseModel):
    """Detection result record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique detection identifier",
    )
    video_id: str = Field(..., min_length=1, description="Associated video ID")
    timestamp: float = Field(..., ge=0, description="Timestamp in video (seconds)")
    frame_number: int = Field(..., ge=0, description="Frame number in video")
    detector_type: str = Field(..., min_length=1, description="Type of detector used")
    bounding_box: BoundingBox = Field(..., description="Detected bounding box")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    area_type: str = Field(..., description="Type of detected area")
    needs_review: bool = Field(
        default=False, description="Whether detection needs manual review"
    )
    reviewed_by: str | None = Field(None, description="User who reviewed detection")
    reviewed_at: datetime | None = Field(None, description="Review timestamp")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    detection_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Detection-specific metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Validate area type against supported types."""
        if v not in SUPPORTED_AREA_TYPES:
            msg = (
                f"Unsupported area type: {v}. Must be one of "
                f"{list(SUPPORTED_AREA_TYPES.keys())}"
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_review_fields(self) -> DetectionRecord:
        """Validate review fields consistency."""
        if self.reviewed_by is not None and self.reviewed_at is None:
            msg = "reviewed_at must be set when reviewed_by is provided"
            raise ValueError(msg)
        if self.reviewed_at is not None and self.reviewed_by is None:
            msg = "reviewed_by must be set when reviewed_at is provided"
            raise ValueError(msg)
        return self

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence."""
        return self.confidence >= TO_REVIEW_CONFIDENCE_THRESHOLD

    @property
    def is_reviewed(self) -> bool:
        """Check if detection has been reviewed."""
        return self.reviewed_by is not None


class TimelineRecord(BaseModel):
    """Timeline record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique timeline identifier",
    )
    video_id: str = Field(..., min_length=1, description="Associated video ID")
    name: str = Field(..., min_length=1, description="Timeline name")
    version: int = Field(default=1, gt=0, description="Timeline version number")
    timeline_data: dict[str, Any] = Field(..., description="Serialized timeline data")
    status: str = Field(default="draft", description="Timeline status")
    created_by: str | None = Field(None, description="User who created timeline")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate timeline status."""
        valid_statuses = ["draft", "approved", "processed", "archived"]
        if v not in valid_statuses:
            msg = f"Invalid status: {v}. Must be one of {valid_statuses}"
            raise ValueError(msg)
        return v

    @field_validator("timeline_data")
    @classmethod
    def validate_timeline_data(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate timeline data structure."""
        required_fields = ["blur_regions", "video_duration"]
        for field in required_fields:
            if field not in v:
                msg = f"Timeline data missing required field: {field}"
                raise ValueError(msg)
        return v

    @property
    def is_draft(self) -> bool:
        """Check if timeline is in draft status."""
        return self.status == "draft"

    @property
    def is_approved(self) -> bool:
        """Check if timeline is approved."""
        return self.status == "approved"

    @property
    def blur_region_count(self) -> int:
        """Get number of blur regions in timeline."""
        return len(self.timeline_data.get("blur_regions", []))


class ProcessingJobRecord(BaseModel):
    """Processing job record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier"
    )
    video_id: str = Field(..., min_length=1, description="Associated video ID")
    timeline_id: str = Field(..., min_length=1, description="Associated timeline ID")
    status: str = Field(default="pending", description="Job status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Job progress percentage"
    )
    output_path: Path | None = Field(None, description="Path to output file")
    error_message: str | None = Field(None, description="Error message if job failed")
    processing_config: dict[str, Any] = Field(
        default_factory=dict, description="Processing configuration"
    )
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate job status."""
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            msg = f"Invalid status: {v}. Must be one of {valid_statuses}"
            raise ValueError(msg)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, v: Path | None) -> Path | None:
        """Convert to Path if provided."""
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def validate_job_consistency(self) -> ProcessingJobRecord:
        """Validate job field consistency."""
        if self.status == "completed" and self.completed_at is None:
            msg = "completed_at must be set for completed jobs"
            raise ValueError(msg)
        if self.status == "failed" and self.error_message is None:
            msg = "error_message should be set for failed jobs"
            raise ValueError(msg)
        if self.status == "running" and self.started_at is None:
            msg = "started_at must be set for running jobs"
            raise ValueError(msg)
        return self

    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in ["pending", "running"]

    @property
    def is_finished(self) -> bool:
        """Check if job has finished (completed, failed, or cancelled)."""
        return self.status in ["completed", "failed", "cancelled"]

    @property
    def duration(self) -> float | None:
        """Calculate job duration in seconds if applicable."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ModelRecord(BaseModel):
    """Trained model record for database storage."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique model identifier"
    )
    name: str = Field(..., min_length=1, description="Model name")
    model_type: str = Field(
        ..., min_length=1, description="Type of model (template, classifier, etc.)"
    )
    area_type: str = Field(..., description="Type of area this model detects")
    version: str = Field(default="1.0", description="Model version")
    file_path: Path = Field(..., description="Path to model file")
    training_data_count: int = Field(
        default=0, ge=0, description="Number of training samples"
    )
    accuracy_metrics: dict[str, float] = Field(
        default_factory=dict, description="Model accuracy metrics"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional model metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Validate area type against supported types."""
        if v not in SUPPORTED_AREA_TYPES:
            supported = list(SUPPORTED_AREA_TYPES.keys())
            msg = f"Unsupported area type: {v}. Must be one of {supported!r}"
            raise ValueError(msg)
        return v

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        """Validate and normalize file path."""
        return Path(v).resolve()

    @field_validator("accuracy_metrics")
    @classmethod
    def validate_accuracy_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate accuracy metric values."""
        for metric_name, value in v.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                msg = f"Accuracy metric '{metric_name}' must be between 0.0 and 1.0"
                raise ValueError(msg)
        return v

    @property
    def has_accuracy_data(self) -> bool:
        """Check if model has accuracy metrics."""
        return len(self.accuracy_metrics) > 0

    @property
    def primary_accuracy(self) -> float | None:
        """Get primary accuracy metric (precision, accuracy, or first available)."""
        for preferred_metric in ["precision", "accuracy", "f1_score"]:
            if preferred_metric in self.accuracy_metrics:
                return self.accuracy_metrics[preferred_metric]
        if self.accuracy_metrics:
            return next(iter(self.accuracy_metrics.values()))
        return None


class TimeRangeAnnotationRecord(BaseModel):
    """Persistent record for a time-range annotation spanning multiple frames."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique time range annotation identifier",
    )
    video_id: str = Field(..., min_length=1, description="Associated video ID")
    start_time: float = Field(..., ge=0, description="Range start (seconds)")
    end_time: float = Field(..., ge=0, description="Range end (seconds)")
    bounding_box: BoundingBox = Field(
        ..., description="Region applied across the range"
    )
    area_type: str = Field(..., description="Type of sensitive area")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    sample_interval: float = Field(
        default=1.0, gt=0, description="Sampling interval in seconds"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Validate area type against supported types."""
        if v not in SUPPORTED_AREA_TYPES:
            msg = (
                f"Unsupported area type: {v}. Must be one of "
                f"{list(SUPPORTED_AREA_TYPES.keys())}"
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_range(self) -> TimeRangeAnnotationRecord:
        """Validate that end_time is greater than start_time."""
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)
        return self
