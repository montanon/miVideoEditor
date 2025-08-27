"""Base detector interface and common detection functionality."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from mivideoeditor.core.models import DetectionResult, SensitiveArea

# Constants for frame validation
MIN_FRAME_CHANNELS = 1
STANDARD_RGB_CHANNELS = 3
RGBA_CHANNELS = 4
VALID_FRAME_CHANNELS = [MIN_FRAME_CHANNELS, STANDARD_RGB_CHANNELS, RGBA_CHANNELS]


class DetectionConfig:
    """Configuration for detection algorithms."""

    def __init__(
        self,
        frame_step: int = 10,
        confidence_threshold: float = 0.7,
        max_regions_per_frame: int = 5,
        min_detection_area: int = 100,
        max_detection_area: int | None = None,
        template_scales: list[float] | None = None,
        *,
        enable_motion_tracking: bool = True,
        enable_temporal_filtering: bool = True,
    ) -> None:
        self.frame_step = frame_step
        self.confidence_threshold = confidence_threshold
        self.max_regions_per_frame = max_regions_per_frame
        self.min_detection_area = min_detection_area
        self.max_detection_area = max_detection_area
        self.enable_motion_tracking = enable_motion_tracking
        self.enable_temporal_filtering = enable_temporal_filtering
        self.template_scales = template_scales or [0.8, 1.0, 1.2]

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.confidence_threshold <= 1.0:
            msg = (
                "confidence_threshold must be in (0, 1], "
                f"got {self.confidence_threshold}"
            )
            raise ValueError(msg)

        if self.frame_step < 1:
            msg = f"frame_step must be >= 1, got {self.frame_step}"
            raise ValueError(msg)

        if self.max_regions_per_frame < 1:
            msg = (
                f"max_regions_per_frame must be >= 1, got {self.max_regions_per_frame}"
            )
            raise ValueError(msg)

        if self.min_detection_area < 1:
            msg = f"min_detection_area must be >= 1, got {self.min_detection_area}"
            raise ValueError(msg)


class BaseDetector(ABC):
    """Abstract base class for all detection algorithms."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.is_trained = False
        self.detection_stats = {
            "total_detections": 0,
            "total_frames_processed": 0,
            "average_detection_time": 0.0,
            "average_confidence": 0.0,
        }

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect sensitive regions in a single frame."""

    @abstractmethod
    def train(self, annotations: list[SensitiveArea]) -> None:
        """Train the detector on annotated data."""

    def batch_detect(
        self, frames: list[np.ndarray], timestamps: list[float] | None = None
    ) -> list[DetectionResult]:
        """Detect regions in multiple frames with optimized batch processing."""
        if timestamps is None:
            timestamps = [float(i) for i in range(len(frames))]

        if len(frames) != len(timestamps):
            msg = (
                "Frames and timestamps length mismatch: "
                f"{len(frames)} != {len(timestamps)}"
            )
            raise ValueError(msg)

        results = []
        for frame, timestamp in zip(frames, timestamps, strict=True):
            result = self.detect(frame, timestamp)
            results.append(result)

        return results

    def get_detection_stats(self) -> dict[str, Any]:
        """Get detector performance statistics."""
        return self.detection_stats.copy()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.detection_stats = {
            "total_detections": 0,
            "total_frames_processed": 0,
            "average_detection_time": 0.0,
            "average_confidence": 0.0,
        }

    def _update_stats(self, detection_result: DetectionResult) -> None:
        """Update internal performance statistics."""
        self.detection_stats["total_frames_processed"] += 1
        self.detection_stats["total_detections"] += len(detection_result.detections)

        # Update running average of detection time
        current_avg = self.detection_stats["average_detection_time"]
        frame_count = self.detection_stats["total_frames_processed"]
        new_avg = (
            current_avg * (frame_count - 1) + detection_result.detection_time
        ) / frame_count
        self.detection_stats["average_detection_time"] = new_avg

        # Update running average of confidence
        if detection_result.detections:
            current_conf_avg = self.detection_stats["average_confidence"]
            frame_confidence = detection_result.average_confidence
            new_conf_avg = (
                current_conf_avg * (frame_count - 1) + frame_confidence
            ) / frame_count
            self.detection_stats["average_confidence"] = new_conf_avg

    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validate input frame format."""
        if frame.size == 0:
            msg = "Input frame is empty"
            raise ValueError(msg)

        if frame.ndim not in [2, 3]:
            msg = f"Frame must be 2D or 3D array, got {frame.ndim}D"
            raise ValueError(msg)

        if frame.ndim == 3 and frame.shape[2] not in VALID_FRAME_CHANNELS:
            msg = (
                f"Frame must have {VALID_FRAME_CHANNELS} channels, got {frame.shape[2]}"
            )
            raise ValueError(msg)

    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        model_data = {
            "config": {
                "frame_step": self.config.frame_step,
                "confidence_threshold": self.config.confidence_threshold,
                "max_regions_per_frame": self.config.max_regions_per_frame,
                "min_detection_area": self.config.min_detection_area,
                "max_detection_area": self.config.max_detection_area,
            },
            "is_trained": self.is_trained,
            "stats": self.detection_stats,
            "detector_type": self.__class__.__name__,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("r", encoding="utf-8") as f:
            model_data = json.load(f)

        # Validate model compatibility
        if model_data.get("detector_type") != self.__class__.__name__:
            msg = (
                f"Model type mismatch: expected {self.__class__.__name__}, "
                f"got {model_data.get('detector_type')}"
            )
            raise ValueError(msg)

        # Load basic state
        self.is_trained = model_data.get("is_trained", False)
        self.detection_stats = model_data.get("stats", self.detection_stats)


class DetectionError(Exception):
    """Base exception for detection-related errors."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class TrainingError(DetectionError):
    """Exception raised during detector training."""


class DetectionTimeoutError(DetectionError):
    """Exception raised when detection takes too long."""


def create_detection_result_empty(timestamp: float = 0.0) -> DetectionResult:
    """Create an empty detection result."""
    return DetectionResult(
        detections=[],
        detection_time=0.0,
        detector_type="empty",
        timestamp=timestamp,
    )


def validate_detection_config(config: DetectionConfig) -> None:
    """Validate detection configuration comprehensively."""
    # Validate basic configuration
    if not 0.0 < config.confidence_threshold <= 1.0:
        msg = (
            f"confidence_threshold must be in (0, 1], got {config.confidence_threshold}"
        )
        raise ValueError(msg)

    if config.frame_step < 1:
        msg = f"frame_step must be >= 1, got {config.frame_step}"
        raise ValueError(msg)

    if config.max_regions_per_frame < 1:
        msg = f"max_regions_per_frame must be >= 1, got {config.max_regions_per_frame}"
        raise ValueError(msg)

    if config.min_detection_area < 1:
        msg = f"min_detection_area must be >= 1, got {config.min_detection_area}"
        raise ValueError(msg)

    # Additional validation
    if config.template_scales and not all(s > 0 for s in config.template_scales):
        msg = "All template scales must be positive"
        raise ValueError(msg)
