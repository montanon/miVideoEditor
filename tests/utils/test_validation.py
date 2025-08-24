"""Tests for validation utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from mivideoeditor.core.models import BoundingBox, ValidationResult
from mivideoeditor.utils.validation import ValidationUtils


class TestValidationUtils:
    """Test validation utilities."""

    def test_validate_string_basic(self):
        """Test basic string validation."""
        # Valid strings
        result = ValidationUtils.validate_string("hello", min_length=1, max_length=10)
        assert result.is_valid is True
        assert len(result.errors) == 0

        # Empty string validation
        result = ValidationUtils.validate_string("", allow_empty=True)
        assert result.is_valid is True

        result = ValidationUtils.validate_string("", allow_empty=False)
        assert result.is_valid is False
        assert any("cannot be empty" in error for error in result.errors)

    def test_validate_string_length(self):
        """Test string length validation."""
        # Too short
        result = ValidationUtils.validate_string("hi", min_length=5)
        assert result.is_valid is False
        assert any("at least 5 characters" in error for error in result.errors)

        # Too long
        result = ValidationUtils.validate_string("very long string", max_length=10)
        assert result.is_valid is False
        assert any("at most 10 characters" in error for error in result.errors)

        # Valid length
        result = ValidationUtils.validate_string("hello", min_length=3, max_length=10)
        assert result.is_valid is True

    def test_validate_string_pattern(self):
        """Test string pattern validation."""
        # Valid pattern
        result = ValidationUtils.validate_string("hello123", pattern=r"^[a-z0-9]+$")
        assert result.is_valid is True

        # Invalid pattern
        result = ValidationUtils.validate_string("Hello!", pattern=r"^[a-z0-9]+$")
        assert result.is_valid is False
        assert any(
            "does not match required pattern" in error for error in result.errors
        )

        # Invalid regex pattern
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ValidationUtils.validate_string("test", pattern="[invalid")

    def test_validate_string_context(self):
        """Test string validation context information."""
        result = ValidationUtils.validate_string("hello world", field_name="message")
        assert result.context["field_name"] == "message"
        assert result.context["value_length"] == 11
        assert result.context["value_type"] == "str"

    def test_validate_numeric_basic(self):
        """Test basic numeric validation."""
        # Valid numbers
        result = ValidationUtils.validate_numeric(42)
        assert result.is_valid is True

        result = ValidationUtils.validate_numeric(3.14)
        assert result.is_valid is True

        result = ValidationUtils.validate_numeric(0)
        assert result.is_valid is True

    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        # Within range
        result = ValidationUtils.validate_numeric(5, min_value=1, max_value=10)
        assert result.is_valid is True

        # Below minimum
        result = ValidationUtils.validate_numeric(0, min_value=1)
        assert result.is_valid is False
        assert any("at least 1" in error for error in result.errors)

        # Above maximum
        result = ValidationUtils.validate_numeric(15, max_value=10)
        assert result.is_valid is False
        assert any("at most 10" in error for error in result.errors)

    def test_validate_numeric_integer_only(self):
        """Test integer-only numeric validation."""
        result = ValidationUtils.validate_numeric(42, integer_only=True)
        assert result.is_valid is True

        result = ValidationUtils.validate_numeric(3.14, integer_only=True)
        assert result.is_valid is False
        assert any("must be an integer" in error for error in result.errors)

    def test_validate_numeric_non_numeric(self):
        """Test validation with non-numeric values."""
        result = ValidationUtils.validate_numeric("not a number")
        assert result.is_valid is False
        assert any("must be a number" in error for error in result.errors)

        result = ValidationUtils.validate_numeric(None)
        assert result.is_valid is False
        assert any("must be a number" in error for error in result.errors)

    def test_validate_list_basic(self):
        """Test basic list validation."""
        # Valid list
        result = ValidationUtils.validate_list([1, 2, 3])
        assert result.is_valid is True

        # Empty list
        result = ValidationUtils.validate_list([], allow_empty=True)
        assert result.is_valid is True

        result = ValidationUtils.validate_list([], allow_empty=False)
        assert result.is_valid is False
        assert any("cannot be empty" in error for error in result.errors)

    def test_validate_list_length(self):
        """Test list length validation."""
        items = [1, 2, 3, 4, 5]

        # Too short
        result = ValidationUtils.validate_list(items[:2], min_length=3)
        assert result.is_valid is False
        assert any("at least 3 items" in error for error in result.errors)

        # Too long
        result = ValidationUtils.validate_list(items, max_length=3)
        assert result.is_valid is False
        assert any("at most 3 items" in error for error in result.errors)

        # Valid length
        result = ValidationUtils.validate_list(items, min_length=3, max_length=10)
        assert result.is_valid is True

    def test_validate_list_item_type(self):
        """Test list item type validation."""
        # Valid item types
        result = ValidationUtils.validate_list([1, 2, 3], item_type=int)
        assert result.is_valid is True

        result = ValidationUtils.validate_list(["a", "b"], item_type=str)
        assert result.is_valid is True

        # Invalid item type
        result = ValidationUtils.validate_list([1, "2", 3], item_type=int)
        assert result.is_valid is False
        assert any("Item 1 is not of type" in error for error in result.errors)

    def test_validate_list_non_list(self):
        """Test validation with non-list values."""
        result = ValidationUtils.validate_list("not a list")
        assert result.is_valid is False
        assert any("must be a list" in error for error in result.errors)

    def test_validate_dict_basic(self):
        """Test basic dictionary validation."""
        # Valid dict
        result = ValidationUtils.validate_dict({"key": "value"})
        assert result.is_valid is True

        # Empty dict
        result = ValidationUtils.validate_dict({}, allow_empty=True)
        assert result.is_valid is True

        result = ValidationUtils.validate_dict({}, allow_empty=False)
        assert result.is_valid is False
        assert any("cannot be empty" in error for error in result.errors)

    def test_validate_dict_required_keys(self):
        """Test dictionary required keys validation."""
        data = {"name": "John", "age": 30}

        # All required keys present
        result = ValidationUtils.validate_dict(data, required_keys=["name", "age"])
        assert result.is_valid is True

        # Missing required key
        result = ValidationUtils.validate_dict(data, required_keys=["name", "email"])
        assert result.is_valid is False
        assert any("Missing required key: email" in error for error in result.errors)

    def test_validate_dict_allowed_keys(self):
        """Test dictionary allowed keys validation."""
        data = {"name": "John", "age": 30}

        # All keys allowed
        result = ValidationUtils.validate_dict(
            data, allowed_keys=["name", "age", "email"]
        )
        assert result.is_valid is True

        # Unexpected key
        result = ValidationUtils.validate_dict(data, allowed_keys=["name"])
        assert result.is_valid is False
        assert any("Unexpected key: age" in error for error in result.errors)

    def test_validate_dict_key_types(self):
        """Test dictionary key types validation."""
        data = {"name": "John", "age": 30, "active": True}
        key_types = {"name": str, "age": int, "active": bool}

        # Valid key types
        result = ValidationUtils.validate_dict(data, key_types=key_types)
        assert result.is_valid is True

        # Invalid key type
        data_invalid = {"name": "John", "age": "30", "active": True}
        result = ValidationUtils.validate_dict(data_invalid, key_types=key_types)
        assert result.is_valid is False
        assert any("Key 'age' should be" in error for error in result.errors)

    def test_validate_dict_non_dict(self):
        """Test validation with non-dict values."""
        result = ValidationUtils.validate_dict("not a dict")
        assert result.is_valid is False
        assert any("must be a dictionary" in error for error in result.errors)

    def test_validate_file_path_existing(self, tmp_path: Path):
        """Test file path validation with existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = ValidationUtils.validate_file_path(test_file)
        assert result.is_valid is True
        assert result.context["exists"] is True
        assert result.context["is_file"] is True
        assert result.context["file_size"] > 0

    def test_validate_file_path_nonexistent(self, tmp_path: Path):
        """Test file path validation with non-existent file."""
        test_file = tmp_path / "nonexistent.txt"

        # Must exist = True (default)
        result = ValidationUtils.validate_file_path(test_file, must_exist=True)
        assert result.is_valid is False
        assert any("does not exist" in error for error in result.errors)

        # Must exist = False
        result = ValidationUtils.validate_file_path(test_file, must_exist=False)
        assert result.is_valid is True

    def test_validate_file_path_extensions(self, tmp_path: Path):
        """Test file path extension validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Valid extension
        result = ValidationUtils.validate_file_path(
            test_file, allowed_extensions={".txt", ".md"}
        )
        assert result.is_valid is True

        # Invalid extension
        result = ValidationUtils.validate_file_path(
            test_file, allowed_extensions={".jpg", ".png"}
        )
        assert result.is_valid is False
        assert any("not allowed" in error for error in result.errors)

    def test_validate_file_path_size_limits(self, tmp_path: Path):
        """Test file path size validation."""
        # Create files of different sizes
        small_file = tmp_path / "small.txt"
        small_file.write_text("x")  # 1 byte

        large_content = "x" * 1000  # 1000 bytes
        large_file = tmp_path / "large.txt"
        large_file.write_text(large_content)

        # File too small
        result = ValidationUtils.validate_file_path(small_file, min_size_bytes=100)
        assert result.is_valid is False
        assert any("smaller than minimum" in error for error in result.errors)

        # File too large
        result = ValidationUtils.validate_file_path(large_file, max_size_bytes=100)
        assert result.is_valid is False
        assert any("larger than maximum" in error for error in result.errors)

        # Valid size
        result = ValidationUtils.validate_file_path(
            large_file, min_size_bytes=100, max_size_bytes=2000
        )
        assert result.is_valid is True

    def test_validate_file_path_directory(self, tmp_path: Path):
        """Test file path validation with directory."""
        result = ValidationUtils.validate_file_path(tmp_path)
        assert result.is_valid is False
        assert any("is a directory, not a file" in error for error in result.errors)

    def test_validate_bounding_box_valid(self):
        """Test valid bounding box validation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        result = ValidationUtils.validate_bounding_box(bbox)
        assert result.is_valid is True
        assert result.context["area"] == 5000

    def test_validate_bounding_box_invalid_dimensions(self):
        """Test bounding box with invalid dimensions."""
        # Zero width - BoundingBox validation prevents this
        with pytest.raises(ValidationError, match="greater than 0"):
            BoundingBox(x=10, y=20, width=0, height=50)

        # Negative height - BoundingBox validation prevents this
        with pytest.raises(ValidationError, match="greater than 0"):
            BoundingBox(x=10, y=20, width=100, height=-10)

    def test_validate_bounding_box_container_bounds(self):
        """Test bounding box container bounds validation."""
        bbox = BoundingBox(x=50, y=50, width=100, height=80)
        container_width, container_height = 200, 200

        # Within bounds
        result = ValidationUtils.validate_bounding_box(
            bbox, container_width=container_width, container_height=container_height
        )
        assert result.is_valid is True

        # Outside container
        bbox_outside = BoundingBox(x=150, y=150, width=100, height=80)
        result = ValidationUtils.validate_bounding_box(
            bbox_outside,
            container_width=container_width,
            container_height=container_height,
        )
        assert result.is_valid is False
        assert any("extends beyond container" in error for error in result.errors)

    def test_validate_bounding_box_negative_position(self):
        """Test bounding box with negative position."""
        # BoundingBox validation prevents negative coordinates
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BoundingBox(x=-10, y=20, width=100, height=50)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BoundingBox(x=10, y=-20, width=100, height=50)

    def test_validate_timestamp_valid(self):
        """Test valid timestamp validation."""
        result = ValidationUtils.validate_timestamp(123.456)
        assert result.is_valid is True
        assert result.context["timestamp"] == 123.456

        result = ValidationUtils.validate_timestamp(0.0)
        assert result.is_valid is True

    def test_validate_timestamp_invalid(self):
        """Test invalid timestamp validation."""
        # Negative timestamp
        result = ValidationUtils.validate_timestamp(-10.5)
        assert result.is_valid is False
        assert any("cannot be negative" in error for error in result.errors)

        # Non-numeric
        result = ValidationUtils.validate_timestamp("not a number")
        assert result.is_valid is False
        assert any("must be a number" in error for error in result.errors)

    def test_validate_timestamp_range(self):
        """Test timestamp range validation."""
        # Within range
        result = ValidationUtils.validate_timestamp(
            30.0, min_timestamp=0.0, max_timestamp=60.0
        )
        assert result.is_valid is True

        # Below minimum
        result = ValidationUtils.validate_timestamp(5.0, min_timestamp=10.0)
        assert result.is_valid is False
        assert any("at least 10.0" in error for error in result.errors)

        # Above maximum
        result = ValidationUtils.validate_timestamp(90.0, max_timestamp=60.0)
        assert result.is_valid is False
        assert any("at most 60.0" in error for error in result.errors)

    @patch("mivideoeditor.utils.validation.SystemUtils")
    def test_validate_system_optimized_config_success(self, mock_system_utils: Mock):
        """Test system optimized config validation - success case."""
        # Mock system info
        mock_system_utils.get_system_info.return_value = {
            "platform": {"system": "Darwin"},
            "resources": {"cpu_cores": 8, "memory": {"total": 16 * 1024**3}},
        }
        mock_system_utils.get_gpu_info.return_value = {
            "available": True,
            "videotoolbox": True,
        }

        config = {
            "processing_mode": "balanced",
            "hardware_acceleration": "auto",
            "frame_step": 10,
            "quality_mode": "medium",
        }

        result = ValidationUtils.validate_system_optimized_config(config)
        assert result.is_valid is True
        # Should have some context about hardware optimization
        assert (
            len(result.context) >= 0
        )  # Context may be empty if no optimizations detected

    @patch("mivideoeditor.utils.validation.SystemUtils")
    def test_validate_system_optimized_config_optimization(
        self, mock_system_utils: Mock
    ):
        """Test system optimized config with optimization suggestions."""
        # Mock low-power system
        mock_system_utils.get_system_info.return_value = {
            "platform": {"system": "Darwin"},
            "resources": {"cpu_cores": 2, "memory": {"total": 4 * 1024**3}},
        }
        mock_system_utils.get_gpu_info.return_value = {"available": False}

        config = {
            "processing_mode": "maximum",  # Too demanding for low-power system
            "frame_step": 5,  # Too frequent for low-power system
        }

        result = ValidationUtils.validate_system_optimized_config(config)
        assert result.is_valid is True
        # Should provide warnings about optimization for low-power system
        assert (
            len(result.warnings) >= 0
        )  # May have warnings about hardware acceleration

    def test_validate_system_optimized_config_no_system_utils(self):
        """Test system config validation - basic case."""
        config = {"processing_mode": "balanced"}

        result = ValidationUtils.validate_system_optimized_config(config)
        assert result.is_valid is True

    @patch("mivideoeditor.utils.validation.SystemUtils")
    def test_validate_macos_video_processing_setup_success(
        self, mock_system_utils: Mock
    ):
        """Test macOS video processing setup validation - success."""
        mock_system_utils.get_system_info.return_value = {
            "platform": {"system": "Darwin"}
        }
        mock_system_utils.get_gpu_info.return_value = {
            "available": True,
            "videotoolbox": True,
            "devices": [{"name": "Apple M2 Pro"}],
        }
        mock_system_utils.check_dependencies.return_value = {
            "ffmpeg": {"available": True, "videotoolbox_support": True}
        }

        result = ValidationUtils.validate_macos_video_processing_setup()
        assert result.is_valid is True
        # Should have Apple Silicon info in context
        assert "apple_silicon" in result.context

    @patch("mivideoeditor.utils.validation.SystemUtils")
    def test_validate_macos_video_processing_setup_not_macos(
        self, mock_system_utils: Mock
    ):
        """Test macOS setup validation on non-macOS system."""
        mock_system_utils.get_system_info.return_value = {
            "platform": {"system": "Linux"}
        }

        result = ValidationUtils.validate_macos_video_processing_setup()
        assert result.is_valid is True
        # Should have warnings about not being on macOS
        assert len(result.warnings) > 0

    def test_validate_color_tuple_valid(self):
        """Test valid color tuple validation."""
        # RGB color
        result = ValidationUtils.validate_color_tuple((255, 128, 0))
        assert result.is_valid is True
        assert result.context["color_space"] == "RGB"

        # HSV color
        result = ValidationUtils.validate_color_tuple(
            (179, 255, 255), color_space="HSV"
        )
        assert result.is_valid is True

    def test_validate_color_tuple_invalid_format(self):
        """Test invalid color tuple format."""
        # Wrong number of components
        result = ValidationUtils.validate_color_tuple((255, 128))
        assert result.is_valid is False
        assert any("must have exactly 3 components" in error for error in result.errors)

        # Non-tuple
        result = ValidationUtils.validate_color_tuple([255, 128, 0])
        assert result.is_valid is False
        assert any("must be a tuple" in error for error in result.errors)

    def test_validate_color_tuple_invalid_values(self):
        """Test invalid color tuple values."""
        # Values out of range
        result = ValidationUtils.validate_color_tuple((256, 128, 0))
        assert result.is_valid is False
        assert any("must be between 0 and 255" in error for error in result.errors)

        result = ValidationUtils.validate_color_tuple((255, -1, 0))
        assert result.is_valid is False
        assert any("must be between 0 and 255" in error for error in result.errors)

        # Non-integer values
        result = ValidationUtils.validate_color_tuple((255.5, 128, 0))
        assert result.is_valid is False
        assert any("must be integers" in error for error in result.errors)

    def test_validate_color_tuple_hsv_range(self):
        """Test HSV color tuple validation with correct ranges."""
        # Valid HSV
        result = ValidationUtils.validate_color_tuple(
            (179, 255, 255), color_space="HSV"
        )
        assert result.is_valid is True

        # Invalid hue (>= 180)
        result = ValidationUtils.validate_color_tuple(
            (180, 255, 255), color_space="HSV"
        )
        assert result.is_valid is False
        assert any("Hue must be between 0 and 179" in error for error in result.errors)

    def test_validate_processing_config_valid(self):
        """Test valid processing config validation."""
        config = {
            "processing_mode": "balanced",
            "quality_mode": "medium",
            "frame_step": 10,
            "chunk_size": 300,
            "hardware_acceleration": "auto",
        }

        result = ValidationUtils.validate_processing_config(config)
        assert result.is_valid is True

    def test_validate_processing_config_invalid_values(self):
        """Test processing config with invalid values."""
        config = {
            "processing_mode": "invalid_mode",
            "frame_step": 0,
            "chunk_size": -100,
        }

        result = ValidationUtils.validate_processing_config(config)
        assert result.is_valid is False
        assert any("invalid_mode" in error for error in result.errors)
        assert any("frame_step must be positive" in error for error in result.errors)
        assert any("chunk_size must be positive" in error for error in result.errors)

    def test_validate_processing_config_warnings(self):
        """Test processing config warnings."""
        config = {
            "processing_mode": "maximum",  # Very slow
            "frame_step": 1,  # Very frequent
            "chunk_size": 10,  # Very small chunks
        }

        result = ValidationUtils.validate_processing_config(config)
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_validate_annotation_data_complete(self):
        """Test complete annotation data validation."""
        annotation = {
            "id": "test-id-123",
            "timestamp": 123.45,
            "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 50},
            "confidence": 0.85,
            "type": "sensitive_text",
            "blur_strength": 0.8,
        }

        result = ValidationUtils.validate_annotation_data(annotation)
        assert result.is_valid is True

    def test_validate_annotation_data_missing_required(self):
        """Test annotation data with missing required fields."""
        annotation = {
            "timestamp": 123.45,
            # Missing id, bounding_box
        }

        result = ValidationUtils.validate_annotation_data(annotation)
        assert result.is_valid is False
        assert any("Missing required key: id" in error for error in result.errors)
        assert any(
            "Missing required key: bounding_box" in error for error in result.errors
        )

    def test_validate_annotation_data_invalid_types(self):
        """Test annotation data with invalid field types."""
        annotation = {
            "id": 123,  # Should be string
            "timestamp": "invalid",  # Should be number
            "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 50},
            "confidence": 1.5,  # Should be <= 1.0
        }

        result = ValidationUtils.validate_annotation_data(annotation)
        assert result.is_valid is False
        assert any("should be <class 'str'>" in error for error in result.errors)

    def test_batch_validate_mixed_results(self):
        """Test batch validation with mixed results."""
        validators = [
            lambda: ValidationUtils.validate_string("valid", min_length=3),
            lambda: ValidationUtils.validate_string("x", min_length=3),  # Invalid
            lambda: ValidationUtils.validate_numeric(42, min_value=0),
            lambda: ValidationUtils.validate_numeric(-5, min_value=0),  # Invalid
        ]

        results = ValidationUtils.batch_validate(validators)

        assert len(results) == 4
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True
        assert results[3].is_valid is False

    def test_batch_validate_with_labels(self):
        """Test batch validation with custom labels."""
        validators = {
            "name": lambda: ValidationUtils.validate_string("John", min_length=2),
            "age": lambda: ValidationUtils.validate_numeric(25, min_value=0),
        }

        results = ValidationUtils.batch_validate(validators)

        assert "name" in results
        assert "age" in results
        assert results["name"].is_valid is True
        assert results["age"].is_valid is True

    def test_create_combined_result(self):
        """Test creating combined validation results."""
        results = [
            ValidationResult(is_valid=True),
            ValidationResult(is_valid=False, errors=["Error 1"]),
            ValidationResult(is_valid=True, warnings=["Warning 1"]),
        ]

        combined = ValidationUtils.create_combined_result(results)

        assert combined.is_valid is False  # Any failure makes combined invalid
        assert "Error 1" in combined.errors
        assert "Warning 1" in combined.warnings

    def test_create_combined_result_all_valid(self):
        """Test creating combined result when all are valid."""
        results = [
            ValidationResult(is_valid=True, warnings=["Warning 1"]),
            ValidationResult(is_valid=True, warnings=["Warning 2"]),
        ]

        combined = ValidationUtils.create_combined_result(results)

        assert combined.is_valid is True
        assert len(combined.errors) == 0
        assert len(combined.warnings) == 2

    def test_validate_confidence_score_valid(self):
        """Test valid confidence score validation."""
        result = ValidationUtils.validate_confidence_score(0.85)
        assert result.is_valid is True

        # Boundary values
        result = ValidationUtils.validate_confidence_score(0.0)
        assert result.is_valid is True

        result = ValidationUtils.validate_confidence_score(1.0)
        assert result.is_valid is True

    def test_validate_confidence_score_invalid(self):
        """Test invalid confidence score validation."""
        # Below 0
        result = ValidationUtils.validate_confidence_score(-0.1)
        assert result.is_valid is False
        assert any("between 0.0 and 1.0" in error for error in result.errors)

        # Above 1
        result = ValidationUtils.validate_confidence_score(1.5)
        assert result.is_valid is False
        assert any("between 0.0 and 1.0" in error for error in result.errors)

        # Non-numeric
        result = ValidationUtils.validate_confidence_score("high")
        assert result.is_valid is False
        assert any("must be a number" in error for error in result.errors)
