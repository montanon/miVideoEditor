"""Validation utilities for data validation across modules."""

from __future__ import annotations

import platform
import re
import uuid
from pathlib import Path
from typing import Any, Union

from mivideoeditor.core.constants import BLUR_FILTER_TYPES, SUPPORTED_AREA_TYPES
from mivideoeditor.core.models import BoundingBox, ValidationResult
from mivideoeditor.utils.system import SystemUtils


class ValidationUtils:
    """Utilities for comprehensive data validation across modules."""

    @staticmethod
    def validate_bounding_box(
        bbox: BoundingBox,
        frame_size: tuple[int, int] | None = None,
        min_area: int = 1,
        max_area: int | None = None,
    ) -> ValidationResult:
        """Comprehensive bounding box validation."""
        result = ValidationResult(is_valid=True)

        # Basic dimension validation (handled by Pydantic, but double-check)
        if bbox.width <= 0 or bbox.height <= 0:
            result.add_error(
                f"Invalid dimensions: {bbox.width}x{bbox.height} (must be > 0)"
            )

        # Coordinate validation
        if bbox.x < 0 or bbox.y < 0:
            result.add_error(f"Negative coordinates: ({bbox.x}, {bbox.y})")

        # Area validation
        area = bbox.area
        if area < min_area:
            result.add_error(f"Area too small: {area} pixels (minimum {min_area})")

        if max_area is not None and area > max_area:
            result.add_error(f"Area too large: {area} pixels (maximum {max_area})")

        # Frame boundary validation
        if frame_size is not None:
            frame_width, frame_height = frame_size

            if bbox.x >= frame_width:
                result.add_error(
                    f"X coordinate {bbox.x} exceeds frame width {frame_width}"
                )
            if bbox.y >= frame_height:
                result.add_error(
                    f"Y coordinate {bbox.y} exceeds frame height {frame_height}"
                )

            if bbox.x + bbox.width > frame_width:
                result.add_error(
                    f"Bounding box extends beyond frame width: "
                    f"{bbox.x + bbox.width} > {frame_width}"
                )
            if bbox.y + bbox.height > frame_height:
                result.add_error(
                    f"Bounding box extends beyond frame height: "
                    f"{bbox.y + bbox.height} > {frame_height}"
                )

            # Aspect ratio warnings for extreme cases
            aspect_ratio = bbox.width / bbox.height
            if aspect_ratio > 10:
                result.add_warning(f"Very wide aspect ratio: {aspect_ratio:.1f}:1")
            elif aspect_ratio < 0.1:
                result.add_warning(f"Very tall aspect ratio: {aspect_ratio:.1f}:1")

        # Reasonable size warnings
        if area < 100:
            result.add_warning(f"Very small area: {area} pixels")

        return result

    @staticmethod
    def validate_confidence(
        confidence: float,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        warning_threshold: float = 0.5,
    ) -> ValidationResult:
        """Validate confidence scores with contextual warnings."""
        result = ValidationResult(is_valid=True)

        # Range validation
        if not (min_confidence <= confidence <= max_confidence):
            result.add_error(
                f"Confidence {confidence} out of range [{min_confidence}, {max_confidence}]"
            )

        # Quality warnings
        if confidence < warning_threshold:
            result.add_warning(f"Low confidence score: {confidence:.3f}")

        # Extreme values
        if confidence == 0.0:
            result.add_warning("Zero confidence - detection may be unreliable")
        elif confidence == 1.0:
            result.add_warning("Perfect confidence - may indicate manual annotation")

        return result

    @staticmethod
    def validate_timestamp(
        timestamp: float,
        video_duration: float | None = None,
        *,
        allow_negative: bool = False,
    ) -> ValidationResult:
        """Validate video timestamp values."""
        result = ValidationResult(is_valid=True)

        # Basic range validation
        if not allow_negative and timestamp < 0:
            result.add_error(f"Negative timestamp not allowed: {timestamp}")

        # Video duration bounds
        if video_duration is not None:
            if timestamp > video_duration:
                result.add_error(
                    f"Timestamp {timestamp} exceeds video duration {video_duration}"
                )

            # Warning for timestamps very close to end
            if video_duration > 0 and timestamp > video_duration * 0.99:
                result.add_warning(
                    f"Timestamp {timestamp} very close to video end ({video_duration})"
                )

        # Precision warnings
        if timestamp != round(timestamp, 3):
            result.add_warning(f"High precision timestamp: {timestamp} (rounded to ms)")

        return result

    @staticmethod
    def validate_time_range(
        start_time: float,
        end_time: float,
        video_duration: float | None = None,
        min_duration: float = 0.1,
    ) -> ValidationResult:
        """Validate time range consistency and bounds."""
        result = ValidationResult(is_valid=True)

        # Order validation
        if end_time <= start_time:
            result.add_error(
                f"End time {end_time} must be greater than start time {start_time}"
            )

        # Duration validation
        duration = end_time - start_time
        if duration < min_duration:
            result.add_error(f"Duration {duration} too short (minimum {min_duration})")

        # Individual timestamp validation
        start_validation = ValidationUtils.validate_timestamp(
            start_time, video_duration
        )
        end_validation = ValidationUtils.validate_timestamp(end_time, video_duration)

        result = result.merge(start_validation).merge(end_validation)

        # Range-specific checks
        if video_duration is not None and duration > video_duration * 0.8:
            result.add_warning(
                f"Long duration {duration} covers "
                f"{duration / video_duration * 100:.1f}% of video"
            )

        return result

    @staticmethod
    def validate_area_type(area_type: str) -> ValidationResult:
        """Validate area type against supported types."""
        result = ValidationResult(is_valid=True)

        if not area_type:
            result.add_error("Area type cannot be empty")
            return result

        if area_type not in SUPPORTED_AREA_TYPES:
            result.add_error(
                f"Unsupported area type: '{area_type}'. "
                f"Supported types: {list(SUPPORTED_AREA_TYPES.keys())}"
            )

        return result

    @staticmethod
    def validate_blur_type(blur_type: str) -> ValidationResult:
        """Validate blur filter type against supported types."""
        result = ValidationResult(is_valid=True)

        if not blur_type:
            result.add_error("Blur type cannot be empty")
            return result

        if blur_type not in BLUR_FILTER_TYPES:
            result.add_error(
                f"Unsupported blur type: '{blur_type}'. "
                f"Supported types: {list(BLUR_FILTER_TYPES.keys())}"
            )

        return result

    @staticmethod
    def validate_file_path(
        file_path: Union[str, Path],
        allowed_extensions: list[str] | None = None,
        max_size_bytes: int | None = None,
        *,
        must_exist: bool = True,
    ) -> ValidationResult:
        """Validate file path and properties."""
        result = ValidationResult(is_valid=True)

        # Convert to Path object
        try:
            path = Path(file_path)
        except (TypeError, ValueError) as e:
            result.add_error(f"Invalid path format: {e}")
            return result

        # Existence check
        if must_exist and not path.exists():
            result.add_error(f"File does not exist: {path}")
            return result

        if path.exists():
            # File type validation
            if not path.is_file():
                result.add_error(f"Path is not a file: {path}")

            # Extension validation
            if allowed_extensions is not None:
                extension = path.suffix.lower()
                if extension not in [ext.lower() for ext in allowed_extensions]:
                    result.add_error(
                        f"File extension '{extension}' not allowed. "
                        f"Allowed: {allowed_extensions}"
                    )

            # Size validation
            if max_size_bytes is not None:
                try:
                    file_size = path.stat().st_size
                    if file_size > max_size_bytes:
                        size_mb = file_size / (1024 * 1024)
                        max_mb = max_size_bytes / (1024 * 1024)
                        result.add_error(
                            f"File too large: {size_mb:.1f}MB (maximum {max_mb:.1f}MB)"
                        )
                except OSError as e:
                    result.add_error(f"Cannot check file size: {e}")

        # Path security validation
        path_str = str(path.resolve())
        if ".." in str(path) or path_str != str(path):
            result.add_warning("Path contains relative components")

        return result

    @staticmethod
    def validate_video_file(file_path: Union[str, Path]) -> ValidationResult:
        """Specialized validation for video files."""
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
        max_size = 50 * 1024 * 1024 * 1024  # 50GB limit

        result = ValidationUtils.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=video_extensions,
            max_size_bytes=max_size,
        )

        if not result.is_valid:
            return result

        # Additional video-specific checks could be added here
        path = Path(file_path)

        # Check for common problematic characters in filename
        filename = path.name
        if re.search(r'[<>:"|?*]', filename):
            result.add_warning("Filename contains potentially problematic characters")

        # Very long filename warning
        if len(filename) > 255:
            result.add_warning(f"Very long filename ({len(filename)} characters)")

        return result

    @staticmethod
    def validate_annotation_data(annotation_data: dict[str, Any]) -> ValidationResult:
        """Validate annotation data dictionary comprehensively."""
        result = ValidationResult(is_valid=True)

        # Required fields
        required_fields = ["timestamp", "bounding_box", "area_type"]
        for field in required_fields:
            if field not in annotation_data:
                result.add_error(f"Missing required field: {field}")

        if not result.is_valid:
            return result

        # Validate timestamp
        try:
            timestamp = float(annotation_data["timestamp"])
            timestamp_result = ValidationUtils.validate_timestamp(timestamp)
            result = result.merge(timestamp_result)
        except (ValueError, TypeError):
            result.add_error(
                f"Invalid timestamp format: {annotation_data['timestamp']}"
            )

        # Validate bounding box
        bbox_data = annotation_data.get("bounding_box", {})
        if isinstance(bbox_data, dict):
            try:
                # Extract bbox parameters
                required_bbox_fields = ["x", "y", "width", "height"]
                for field in required_bbox_fields:
                    if field not in bbox_data:
                        result.add_error(f"Missing bounding box field: {field}")

                if result.is_valid:
                    bbox = BoundingBox(**bbox_data)
                    bbox_result = ValidationUtils.validate_bounding_box(bbox)
                    result = result.merge(bbox_result)

            except (ValueError, TypeError, AttributeError) as e:
                result.add_error(f"Invalid bounding box data: {e}")
        else:
            result.add_error("Bounding box must be a dictionary")

        # Validate area type
        area_type = annotation_data.get("area_type", "")
        area_type_result = ValidationUtils.validate_area_type(area_type)
        result = result.merge(area_type_result)

        # Validate optional confidence
        if "confidence" in annotation_data:
            try:
                confidence = float(annotation_data["confidence"])
                confidence_result = ValidationUtils.validate_confidence(confidence)
                result = result.merge(confidence_result)
            except (ValueError, TypeError):
                result.add_error(
                    f"Invalid confidence format: {annotation_data['confidence']}"
                )

        # Validate optional metadata
        if "metadata" in annotation_data:
            metadata = annotation_data["metadata"]
            if not isinstance(metadata, dict):
                result.add_warning("Metadata should be a dictionary")
            elif len(str(metadata)) > 10000:  # Arbitrary large size limit
                result.add_warning("Very large metadata (>10KB)")

        return result

    @staticmethod
    def validate_processing_config(config_data: dict[str, Any]) -> ValidationResult:
        """Validate processing configuration data."""
        result = ValidationResult(is_valid=True)

        # Quality mode validation
        quality_mode = config_data.get("quality_mode", "balanced")
        valid_modes = ["fast", "balanced", "high", "maximum"]
        if quality_mode not in valid_modes:
            result.add_error(
                f"Invalid quality mode: {quality_mode}. Valid: {valid_modes}"
            )

        # Frame step validation
        if "frame_step" in config_data:
            try:
                frame_step = int(config_data["frame_step"])
                if frame_step < 1:
                    result.add_error(f"Frame step must be >= 1, got: {frame_step}")
                elif frame_step > 300:  # 10 seconds at 30fps
                    result.add_warning(f"Very large frame step: {frame_step}")
            except (ValueError, TypeError):
                result.add_error(f"Invalid frame step: {config_data['frame_step']}")

        # Confidence threshold validation
        if "confidence_threshold" in config_data:
            try:
                threshold = float(config_data["confidence_threshold"])
                threshold_result = ValidationUtils.validate_confidence(threshold)
                result = result.merge(threshold_result)
            except (ValueError, TypeError):
                result.add_error(
                    f"Invalid confidence threshold: {config_data['confidence_threshold']}"
                )

        # Blur strength validation
        if "blur_strength" in config_data:
            try:
                strength = float(config_data["blur_strength"])
                if not (0.0 <= strength <= 2.0):
                    result.add_error(f"Blur strength must be 0.0-2.0, got: {strength}")
            except (ValueError, TypeError):
                result.add_error(
                    f"Invalid blur strength: {config_data['blur_strength']}"
                )

        # Hardware acceleration validation
        if "hardware_acceleration" in config_data:
            acceleration = config_data["hardware_acceleration"]
            valid_accelerations = ["none", "auto", "cuda", "videotoolbox", "qsv"]
            if acceleration not in valid_accelerations:
                result.add_error(f"Invalid hardware acceleration: {acceleration}")

        return result

    @staticmethod
    def validate_batch_data(
        batch_data: list[dict[str, Any]], max_batch_size: int = 1000
    ) -> ValidationResult:
        """Validate a batch of data items."""
        result = ValidationResult(is_valid=True)

        # Batch size validation
        if not isinstance(batch_data, list):
            result.add_error("Batch data must be a list")
            return result

        batch_size = len(batch_data)
        if batch_size == 0:
            result.add_warning("Empty batch")
            return result

        if batch_size > max_batch_size:
            result.add_error(
                f"Batch too large: {batch_size} items (maximum {max_batch_size})"
            )

        # Individual item validation
        valid_items = 0
        total_errors = 0
        total_warnings = 0

        for i, item in enumerate(batch_data):
            if not isinstance(item, dict):
                result.add_error(f"Item {i} is not a dictionary")
                continue

            # Could add specific item validation here based on item type
            # For now, just check basic structure
            if len(item) == 0:
                result.add_warning(f"Item {i} is empty")
            else:
                valid_items += 1

        # Summary
        result.context.update(
            {
                "batch_size": batch_size,
                "valid_items": valid_items,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
            }
        )

        if valid_items == 0:
            result.add_error("No valid items in batch")
        elif valid_items < batch_size * 0.5:
            result.add_warning(
                f"Many invalid items: only {valid_items}/{batch_size} valid"
            )

        return result

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Remove/replace problematic characters
        sanitized = re.sub(r'[<>:"|?*\\\/]', "_", filename)

        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)

        # Handle multiple dots and spaces
        sanitized = re.sub(r"\.+", ".", sanitized)  # Multiple dots to single
        sanitized = re.sub(r"\s+", " ", sanitized)  # Multiple spaces to single
        sanitized = sanitized.strip(" .")  # Remove leading/trailing spaces and dots

        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed"

        # Truncate if too long
        if len(sanitized) > max_length:
            name, ext = (
                sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            )
            available_length = max_length - len(ext) - 1 if ext else max_length
            sanitized = name[:available_length] + ("." + ext if ext else "")

        return sanitized

    @staticmethod
    def validate_uuid_format(uuid_string: str) -> ValidationResult:
        """Validate UUID string format."""
        result = ValidationResult(is_valid=True)

        try:
            uuid_obj = uuid.UUID(uuid_string)
            # Verify the string representation matches (catches some malformed UUIDs)
            if str(uuid_obj) != uuid_string.lower():
                result.add_warning("UUID string formatting normalized")
        except ValueError as e:
            result.add_error(f"Invalid UUID format: {e}")

        return result

    @staticmethod
    def validate_system_optimized_config(
        config_data: dict[str, Any],
    ) -> ValidationResult:
        """Validate configuration with system-aware optimizations for macOS."""
        result = ValidationResult(is_valid=True)

        # Start with basic validation
        basic_validation = ValidationUtils.validate_processing_config(config_data)
        result = result.merge(basic_validation)

        # Add system-aware enhancements
        try:
            # Get system information
            gpu_info = SystemUtils.get_gpu_info()
            deps = SystemUtils.check_dependencies()

            # macOS-specific optimizations
            if platform.system() == "Darwin":
                # Check for Apple Silicon
                apple_gpu = any(
                    device.get("vendor") == "apple"
                    for device in gpu_info.get("devices", [])
                )

                if apple_gpu:
                    # VideoToolbox recommendations for Apple Silicon
                    hardware_accel = config_data.get("hardware_acceleration", "none")
                    if hardware_accel == "none":
                        result.add_warning(
                            "Consider enabling 'videotoolbox' hardware acceleration for "
                            "optimal performance on Apple Silicon"
                        )
                    elif hardware_accel == "videotoolbox":
                        # Validate VideoToolbox is actually available
                        ffmpeg_hwaccel = deps.get("ffmpeg", {}).get(
                            "hardware_acceleration", {}
                        )
                        if ffmpeg_hwaccel.get("videotoolbox", False):
                            result.context["hardware_acceleration_optimal"] = True
                        else:
                            result.add_error(
                                "VideoToolbox requested but not available in FFmpeg installation"
                            )

            # FFmpeg encoder validation
            ffmpeg_info = deps.get("ffmpeg", {})
            if ffmpeg_info.get("available", False):
                encoders = ffmpeg_info.get("encoders", [])
                if "h264_videotoolbox" in encoders and "hevc_videotoolbox" in encoders:
                    result.context["hardware_encoders_available"] = True

        except ImportError:
            result.add_warning("System utilities not available for optimization checks")
        except (OSError, RuntimeError, AttributeError) as e:
            result.add_warning(f"System validation failed: {e}")

        return result

    @staticmethod
    def validate_macos_video_processing_setup() -> ValidationResult:
        """Comprehensive validation specifically for macOS video processing setup."""
        result = ValidationResult(is_valid=True)

        try:
            if platform.system() != "Darwin":
                result.add_error("This validation is specific to macOS systems")
                return result

            # Check system requirements
            system_validation = SystemUtils.validate_system_requirements()
            if system_validation.get("overall_status") == "fail":
                result.add_error("System fails basic requirements")
                for issue in system_validation.get("critical_issues", []):
                    result.add_error(issue)

            # Check for optimal macOS configuration
            gpu_info = SystemUtils.get_gpu_info()
            deps = SystemUtils.check_dependencies()

            # Apple Silicon specific checks
            apple_gpu = any(
                device.get("vendor") == "apple"
                for device in gpu_info.get("devices", [])
            )

            if apple_gpu:
                result.context["apple_silicon"] = True

                # Check VideoToolbox encoders
                ffmpeg_info = deps.get("ffmpeg", {})
                encoders = ffmpeg_info.get("encoders", [])

                videotoolbox_encoders = [
                    enc for enc in encoders if "videotoolbox" in enc
                ]
                if videotoolbox_encoders:
                    result.context["videotoolbox_encoders"] = videotoolbox_encoders
                    result.add_warning(
                        "Optimal VideoToolbox hardware acceleration available"
                    )
                else:
                    result.add_warning("FFmpeg VideoToolbox encoders not found")
            else:
                result.context["apple_silicon"] = False
                result.add_warning(
                    "Intel Mac detected. Apple Silicon recommended for optimal performance"
                )

        except (OSError, RuntimeError, ImportError) as e:
            result.add_error(f"macOS validation failed: {e}")

        return result
