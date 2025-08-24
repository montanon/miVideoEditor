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
        container_width: int | None = None,
        container_height: int | None = None,
        frame_size: tuple[int, int] | None = None,
        min_area: int = 1,
        max_area: int | None = None,
    ) -> ValidationResult:
        """Comprehensive bounding box validation."""
        result = ValidationResult(is_valid=True)

        # Add context information
        area = bbox.area
        result.context["area"] = area
        result.context["width"] = bbox.width
        result.context["height"] = bbox.height

        # Basic dimension validation
        if bbox.width <= 0:
            result.add_error("Width must be positive")

        if bbox.height <= 0:
            result.add_error("Height must be positive")

        # Coordinate validation
        if bbox.x < 0:
            result.add_error("X coordinate must be non-negative")

        if bbox.y < 0:
            result.add_error("Y coordinate must be non-negative")

        # Container boundary validation (using either container_width/height or frame_size)
        if container_width is not None and container_height is not None:
            if bbox.x + bbox.width > container_width:
                result.add_error("Bounding box extends beyond container")
            if bbox.y + bbox.height > container_height:
                result.add_error("Bounding box extends beyond container")

        elif frame_size is not None:
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

        # Area validation
        if area < min_area:
            result.add_error(f"Area too small: {area} pixels (minimum {min_area})")

        if max_area is not None and area > max_area:
            result.add_error(f"Area too large: {area} pixels (maximum {max_area})")

        # Aspect ratio warnings for extreme cases
        if bbox.width > 0 and bbox.height > 0:
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
        timestamp: Any,
        min_timestamp: float | None = None,
        max_timestamp: float | None = None,
        video_duration: float | None = None,
        *,
        allow_negative: bool = False,
    ) -> ValidationResult:
        """Validate video timestamp values."""
        result = ValidationResult(is_valid=True)

        # Add context information
        result.context["timestamp"] = timestamp

        # Type validation
        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError):
            result.add_error(
                f"Timestamp must be a number, got {type(timestamp).__name__}"
            )
            return result

        # Basic range validation
        if not allow_negative and timestamp < 0:
            result.add_error("Timestamp cannot be negative")

        # Min/max timestamp validation
        if min_timestamp is not None and timestamp < min_timestamp:
            result.add_error(f"Timestamp must be at least {min_timestamp}")

        if max_timestamp is not None and timestamp > max_timestamp:
            result.add_error(f"Timestamp must be at most {max_timestamp}")

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
        allowed_extensions: set[str] | None = None,
        max_size_bytes: int | None = None,
        min_size_bytes: int | None = None,
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

        # Add context information
        exists = path.exists()
        result.context["exists"] = exists

        if exists:
            is_file = path.is_file()
            result.context["is_file"] = is_file

            if is_file:
                try:
                    file_size = path.stat().st_size
                    result.context["file_size"] = file_size
                except OSError:
                    result.context["file_size"] = 0

        # Existence check
        if must_exist and not exists:
            result.add_error("File does not exist")
            return result

        if exists:
            # File type validation
            if not path.is_file():
                result.add_error("Path is a directory, not a file")
                return result

            # Extension validation
            if allowed_extensions is not None:
                extension = path.suffix.lower()
                if extension not in {ext.lower() for ext in allowed_extensions}:
                    result.add_error(f"Extension '{extension}' not allowed")

            # Size validation
            try:
                file_size = path.stat().st_size

                if min_size_bytes is not None and file_size < min_size_bytes:
                    result.add_error(
                        f"File is smaller than minimum {min_size_bytes} bytes"
                    )

                if max_size_bytes is not None and file_size > max_size_bytes:
                    result.add_error(
                        f"File is larger than maximum {max_size_bytes} bytes"
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
        required_fields = ["id", "timestamp", "bounding_box"]
        for field in required_fields:
            if field not in annotation_data:
                result.add_error(f"Missing required key: {field}")

        if not result.is_valid:
            return result

        # Validate id field type
        if "id" in annotation_data:
            id_value = annotation_data["id"]
            if not isinstance(id_value, str):
                result.add_error(
                    f"Key 'id' should be <class 'str'>, got {type(id_value)}"
                )

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

        # Validate area type (if present)
        if "type" in annotation_data:
            area_type = annotation_data["type"]
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

        # Validate optional blur strength
        if "blur_strength" in annotation_data:
            try:
                strength = float(annotation_data["blur_strength"])
                if not 0.0 <= strength <= 2.0:
                    result.add_error(
                        f"Blur strength must be between 0.0 and 2.0, got {strength}"
                    )
            except (ValueError, TypeError):
                result.add_error(
                    f"Invalid blur strength: {annotation_data['blur_strength']}"
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

        # Processing mode validation
        if "processing_mode" in config_data:
            processing_mode = config_data["processing_mode"]
            valid_processing_modes = ["fast", "balanced", "maximum"]
            if processing_mode not in valid_processing_modes:
                result.add_error(
                    f"Invalid processing mode: {processing_mode}. Valid: {valid_processing_modes}"
                )

        # Quality mode validation
        quality_mode = config_data.get("quality_mode", "balanced")
        valid_modes = ["fast", "balanced", "medium", "high", "maximum"]
        if quality_mode not in valid_modes:
            result.add_error(
                f"Invalid quality mode: {quality_mode}. Valid: {valid_modes}"
            )

        # Chunk size validation
        if "chunk_size" in config_data:
            try:
                chunk_size = int(config_data["chunk_size"])
                if chunk_size <= 0:
                    result.add_error("chunk_size must be positive")
                elif chunk_size < 30:  # Less than 1 second at 30fps
                    result.add_warning(f"Very small chunk size: {chunk_size}")
            except (ValueError, TypeError):
                result.add_error(f"Invalid chunk size: {config_data['chunk_size']}")

        # Frame step validation
        if "frame_step" in config_data:
            try:
                frame_step = int(config_data["frame_step"])
                if frame_step < 1:
                    result.add_error("frame_step must be positive")
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
            result.add_warning("System utilities not available")
        except (OSError, RuntimeError, AttributeError) as e:
            result.add_warning(f"System validation failed: {e}")

        return result

    @staticmethod
    def validate_string(
        value: Any,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        context: str | None = None,
        field_name: str | None = None,
        *,
        allow_empty: bool = False,
    ) -> ValidationResult:
        """Validate a string value with various constraints."""
        result = ValidationResult(is_valid=True)

        # Add context information
        if field_name:
            result.context["field_name"] = field_name
        result.context["value_type"] = type(value).__name__
        if isinstance(value, str):
            result.context["value_length"] = len(value)

        # Type validation
        if not isinstance(value, str):
            result.add_error(f"Value must be a string, got {type(value).__name__}")
            return result

        # Empty string validation
        if not value and not allow_empty:
            result.add_error("String cannot be empty")
            return result

        # Length validation
        if min_length is not None and len(value) < min_length:
            result.add_error(f"String must be at least {min_length} characters long")

        if max_length is not None and len(value) > max_length:
            result.add_error(f"String must be at most {max_length} characters long")

        # Pattern validation
        if pattern is not None:
            try:
                if not re.match(pattern, value):
                    result.add_error(
                        f"String does not match required pattern: {pattern}"
                    )
            except re.error as e:
                msg = f"Invalid regex pattern: {e}"
                raise ValueError(msg) from e

        # Context-specific validation
        if context == "filename":
            if re.search(r'[<>:"|?*\\\/]', value):
                result.add_error("String contains invalid filename characters")
        elif context == "identifier":
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
                result.add_error("String is not a valid identifier")

        return result

    @staticmethod
    def validate_numeric(
        value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        *,
        integer_only: bool = False,
        allow_zero: bool = True,
        allow_negative: bool = True,
    ) -> ValidationResult:
        """Validate a numeric value with various constraints."""
        result = ValidationResult(is_valid=True)

        # Type validation
        try:
            if integer_only:
                if not isinstance(value, int) or isinstance(value, bool):
                    # Check if it's a float that's not a whole number
                    if isinstance(value, float):
                        if value != int(value):
                            result.add_error(
                                f"Value must be an integer, got float: {value}"
                            )
                            return result
                        value = int(value)
                    else:
                        result.add_error(
                            f"Value must be an integer, got {type(value).__name__}"
                        )
                        return result
            else:
                value = float(value)
        except (ValueError, TypeError):
            result.add_error(f"Value must be a number, got {type(value).__name__}")
            return result

        # Range validation
        if min_value is not None and value < min_value:
            result.add_error(f"Value must be at least {min_value}")

        if max_value is not None and value > max_value:
            result.add_error(f"Value must be at most {max_value}")

        # Zero validation
        if not allow_zero and value == 0:
            result.add_error("Value cannot be zero")

        # Negative validation
        if not allow_negative and value < 0:
            result.add_error("Value cannot be negative")

        return result

    @staticmethod
    def validate_list(
        value: Any,
        min_length: int | None = None,
        max_length: int | None = None,
        item_type: type | None = None,
        *,
        allow_empty: bool = True,
    ) -> ValidationResult:
        """Validate a list with various constraints."""
        result = ValidationResult(is_valid=True)

        # Type validation
        if not isinstance(value, list):
            result.add_error(f"Value must be a list, got {type(value).__name__}")
            return result

        # Empty list validation
        if not value and not allow_empty:
            result.add_error("List cannot be empty")
            return result

        # Length validation
        if min_length is not None and len(value) < min_length:
            result.add_error(f"List must have at least {min_length} items")

        if max_length is not None and len(value) > max_length:
            result.add_error(f"List must have at most {max_length} items")

        # Item type validation
        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    result.add_error(
                        f"Item {i} is not of type {item_type.__name__}, "
                        f"got {type(item).__name__}"
                    )

        return result

    @staticmethod
    def validate_dict(
        value: Any,
        required_keys: list[str] | None = None,
        allowed_keys: list[str] | None = None,
        key_types: dict[str, type] | None = None,
        *,
        allow_empty: bool = True,
    ) -> ValidationResult:
        """Validate a dictionary with various constraints."""
        result = ValidationResult(is_valid=True)

        # Type validation
        if not isinstance(value, dict):
            result.add_error(f"Value must be a dictionary, got {type(value).__name__}")
            return result

        # Empty dict validation
        if not value and not allow_empty:
            result.add_error("Dictionary cannot be empty")
            return result

        # Required keys validation
        if required_keys:
            for key in required_keys:
                if key not in value:
                    result.add_error(f"Missing required key: {key}")

        # Allowed keys validation
        if allowed_keys:
            for key in value:
                if key not in allowed_keys:
                    result.add_error(f"Unexpected key: {key}")

        # Key type validation
        if key_types:
            for key, expected_type in key_types.items():
                if key in value and not isinstance(value[key], expected_type):
                    result.add_error(
                        f"Key '{key}' should be {expected_type.__name__}, "
                        f"got {type(value[key]).__name__}"
                    )

        return result

    @staticmethod
    def validate_color_tuple(
        value: Any,
        channels: int = 3,
        max_value: int = 255,
        color_space: str = "RGB",
    ) -> ValidationResult:
        """Validate a color tuple (RGB, BGR, HSV, etc.)."""
        result = ValidationResult(is_valid=True)

        # Add context information
        result.context["color_space"] = color_space
        result.context["channels"] = channels
        result.context["max_value"] = max_value

        # Type validation - must be tuple, not list
        if not isinstance(value, tuple):
            result.add_error(f"Color must be a tuple, got {type(value).__name__}")
            return result

        # Channel count validation
        if len(value) != channels:
            result.add_error(
                f"Color must have exactly {channels} components, got {len(value)}"
            )
            return result

        # Value validation based on color space
        if color_space.upper() in ["RGB", "BGR"]:
            for i, channel in enumerate(value):
                if not isinstance(channel, int):
                    result.add_error(
                        f"Color components must be integers, got {type(channel).__name__}"
                    )
                    return result
                elif not 0 <= channel <= max_value:
                    result.add_error(
                        f"Color components must be between 0 and {max_value}, got {channel}"
                    )
                    return result
        elif color_space.upper() == "HSV":
            # Hue: 0-179 (OpenCV standard), Saturation: 0-255, Value: 0-255
            if not 0 <= value[0] <= 179:
                result.add_error(f"Hue must be between 0 and 179, got {value[0]}")
            if not 0 <= value[1] <= 255:
                result.add_error("Saturation must be between 0 and 255")
            if not 0 <= value[2] <= 255:
                result.add_error("Value must be between 0 and 255")

        return result

    @staticmethod
    def validate_confidence_score(
        value: Any,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
    ) -> ValidationResult:
        """Validate a confidence score (typically 0.0 to 1.0)."""
        result = ValidationResult(is_valid=True)

        # Type validation
        try:
            score = float(value)
        except (ValueError, TypeError):
            result.add_error(
                f"Confidence score must be a number, got {type(value).__name__}"
            )
            return result

        # Range validation
        if not min_threshold <= score <= max_threshold:
            result.add_error(
                f"Confidence score must be between {min_threshold} and {max_threshold}, "
                f"got {score}"
            )

        # Warning for extreme values
        if score == 0.0:
            result.add_warning("Confidence score is 0.0 (no confidence)")
        elif score == 1.0:
            result.add_warning("Confidence score is 1.0 (perfect confidence)")
        elif score < 0.3:
            result.add_warning(f"Low confidence score: {score}")

        return result

    @staticmethod
    def batch_validate(validators):
        """Validate a batch of validators (functions that return ValidationResults)."""
        if isinstance(validators, dict):
            # Handle dictionary of validators
            results = {}
            for label, validator in validators.items():
                try:
                    result = validator()
                    result.context["label"] = label
                except Exception as e:
                    result = ValidationResult(is_valid=False)
                    result.add_error(f"Validation failed for {label}: {e}")
                results[label] = result
            return results

        results = []
        for i, validator in enumerate(validators):
            label = f"Item {i}"
            try:
                result = validator()
                result.context["label"] = label
            except Exception as e:
                result = ValidationResult(is_valid=False)
                result.add_error(f"Validation failed for {label}: {e}")
            results.append(result)
        return results

    @staticmethod
    def create_combined_result(
        results: list[ValidationResult],
        *,
        require_all_valid: bool = True,
    ) -> ValidationResult:
        """Combine multiple ValidationResults into a single result."""
        combined = ValidationResult(is_valid=True)

        valid_count = 0
        total_count = len(results)

        for i, result in enumerate(results):
            if result.is_valid:
                valid_count += 1

            # Merge errors and warnings (preserve original messages)
            for error in result.errors:
                combined.add_error(error)

            for warning in result.warnings:
                combined.add_warning(warning)

        # Update validity based on requirement
        if require_all_valid:
            combined.is_valid = valid_count == total_count
        else:
            combined.is_valid = valid_count > 0

        # Add summary to context
        combined.context["valid_count"] = valid_count
        combined.context["total_count"] = total_count
        combined.context["invalid_count"] = total_count - valid_count

        return combined

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
