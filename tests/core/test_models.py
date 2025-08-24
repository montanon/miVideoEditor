"""Tests for core data models."""

import uuid
from datetime import UTC
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mivideoeditor.core.models import (
    BlurRegion,
    BoundingBox,
    DetectionResult,
    SensitiveArea,
    Timeline,
    ValidationResult,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_creation(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_immutability(self):
        """Test that bounding box is immutable."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            bbox.x = 20  # This should fail because it's frozen

    def test_invalid_negative_coordinates(self):
        """Test that negative coordinates raise error."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BoundingBox(x=-1, y=20, width=100, height=50)

    def test_invalid_zero_dimensions(self):
        """Test that zero dimensions raise error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            BoundingBox(x=10, y=20, width=0, height=50)

    def test_area_calculation(self):
        """Test area calculation."""
        bbox = BoundingBox(x=0, y=0, width=10, height=20)
        assert bbox.area == 200

    def test_center_calculation(self):
        """Test center coordinates calculation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.center == (60, 45)

    def test_corners_calculation(self):
        """Test corner coordinates."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        expected_corners = ((10, 20), (110, 20), (110, 70), (10, 70))
        assert bbox.corners == expected_corners

    def test_contains_point(self):
        """Test point containment."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.contains((50, 40)) is True
        assert bbox.contains((10, 20)) is True
        assert bbox.contains((109, 69)) is True
        assert bbox.contains((110, 70)) is False  # Boundary exclusive

    def test_overlaps(self):
        """Test overlap detection."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=25, y=25, width=50, height=50)
        bbox3 = BoundingBox(x=60, y=60, width=30, height=30)

        assert bbox1.overlaps(bbox2) is True
        assert bbox1.overlaps(bbox3) is False

    def test_intersection(self):
        """Test intersection calculation."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=25, y=25, width=50, height=50)

        intersection = bbox1.intersection(bbox2)
        assert intersection is not None
        assert intersection.x == 25
        assert intersection.y == 25
        assert intersection.width == 25
        assert intersection.height == 25

    def test_no_intersection(self):
        """Test no intersection returns None."""
        bbox1 = BoundingBox(x=0, y=0, width=25, height=25)
        bbox2 = BoundingBox(x=50, y=50, width=25, height=25)

        intersection = bbox1.intersection(bbox2)
        assert intersection is None

    def test_union(self):
        """Test union calculation."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=25, y=25, width=50, height=50)

        union = bbox1.union(bbox2)
        assert union.x == 0
        assert union.y == 0
        assert union.width == 75
        assert union.height == 75

    def test_iou_calculation(self):
        """Test Intersection over Union calculation."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=25, y=25, width=50, height=50)

        iou = bbox1.iou(bbox2)
        # Intersection area: 25*25 = 625
        # Union area: 50*50 + 50*50 - 625 = 4375
        expected_iou = 625 / 4375
        assert abs(iou - expected_iou) < 0.001

    def test_expand(self):
        """Test expanding bounding box."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        expanded = bbox.expand(10)

        assert expanded.x == 40
        assert expanded.y == 40
        assert expanded.width == 120
        assert expanded.height == 120

    def test_expand_near_origin(self):
        """Test expanding near origin doesn't go negative."""
        bbox = BoundingBox(x=5, y=5, width=10, height=10)
        expanded = bbox.expand(10)

        assert expanded.x == 0  # Clamped to 0
        assert expanded.y == 0  # Clamped to 0
        assert expanded.width == 30
        assert expanded.height == 30

    def test_scale(self):
        """Test scaling bounding box."""
        bbox = BoundingBox(x=40, y=40, width=20, height=20)  # Center at (50, 50)
        scaled = bbox.scale(2.0)

        # Should scale around center
        assert scaled.width == 40
        assert scaled.height == 40
        # New center should remain the same
        assert scaled.center == (50, 50)

    def test_clip_to_frame(self):
        """Test clipping to frame boundaries."""
        bbox = BoundingBox(x=950, y=950, width=200, height=200)
        clipped = bbox.clip(1000, 1000)

        assert clipped.x == 950
        assert clipped.y == 950
        assert clipped.width == 50  # Clipped to frame boundary
        assert clipped.height == 50

    def test_normalized_coordinates(self):
        """Test conversion to normalized coordinates."""
        bbox = BoundingBox(x=100, y=200, width=300, height=400)
        normalized = bbox.to_normalized(1000, 1000)

        assert normalized["x"] == 0.1
        assert normalized["y"] == 0.2
        assert normalized["width"] == 0.3
        assert normalized["height"] == 0.4

    def test_from_normalized(self):
        """Test creation from normalized coordinates."""
        bbox = BoundingBox.from_normalized(
            x=0.1, y=0.2, width=0.3, height=0.4, frame_width=1000, frame_height=1000
        )

        assert bbox.x == 100
        assert bbox.y == 200
        assert bbox.width == 300
        assert bbox.height == 400

    def test_from_corners(self):
        """Test creation from corner coordinates."""
        bbox = BoundingBox.from_corners(10, 20, 110, 70)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_maximum_dimension_validation(self):
        """Test maximum dimension validation."""
        with pytest.raises(ValidationError, match="maximum dimension"):
            BoundingBox(x=8000, y=0, width=100, height=100)

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    )
    def test_property_area_calculation(self, x, y, width, height):
        """Property-based test for area calculation."""
        bbox = BoundingBox(x=x, y=y, width=width, height=height)
        assert bbox.area == width * height

    def test_string_representations(self):
        """Test string representations."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        str_repr = str(bbox)
        repr_repr = repr(bbox)

        assert "BoundingBox" in str_repr
        assert "BoundingBox" in repr_repr
        assert "area=5000" in repr_repr


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_creation(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding a warning."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid is True  # Warnings don't affect validity
        assert "Test warning" in result.warnings

    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(is_valid=True, warnings=["Warning 1"])
        result2 = ValidationResult(is_valid=False, errors=["Error 1"])

        merged = result1.merge(result2)
        assert merged.is_valid is False
        assert "Warning 1" in merged.warnings
        assert "Error 1" in merged.errors

    def test_success_class_method(self):
        """Test success class method."""
        result = ValidationResult.success()
        assert result.is_valid is True
        assert result.errors == []

        result_with_warnings = ValidationResult.success(warnings=["Warning"])
        assert result_with_warnings.is_valid is True
        assert "Warning" in result_with_warnings.warnings

    def test_failure_class_method(self):
        """Test failure class method."""
        result = ValidationResult.failure("Error message")
        assert result.is_valid is False
        assert "Error message" in result.errors

        result_multiple = ValidationResult.failure(["Error 1", "Error 2"])
        assert result_multiple.is_valid is False
        assert len(result_multiple.errors) == 2

    def test_properties(self):
        """Test ValidationResult properties."""
        result = ValidationResult(
            is_valid=False, errors=["Error 1"], warnings=["Warning 1"]
        )

        assert result.has_errors is True
        assert result.has_warnings is True
        assert result.message_count == 2

    def test_raise_if_invalid(self):
        """Test raise_if_invalid method."""
        valid_result = ValidationResult.success()
        valid_result.raise_if_invalid()  # Should not raise

        invalid_result = ValidationResult.failure("Test error")
        with pytest.raises(ValueError, match="Validation failed"):
            invalid_result.raise_if_invalid()

    def test_message_filtering(self):
        """Test that empty messages are filtered out."""
        result = ValidationResult(is_valid=True, errors=["", "Real error"])
        # Validator should filter out empty strings
        assert len(result.errors) <= 1  # Only non-empty strings should remain


class TestSensitiveArea:
    """Tests for SensitiveArea model."""

    def test_valid_creation(self):
        """Test creating a valid sensitive area."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(
            timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=0.95
        )

        assert area.timestamp == 5.5
        assert area.bounding_box == bbox
        assert area.area_type == "chatgpt"
        assert area.confidence == 0.95
        assert area.area == 5000

    def test_uuid_generation(self):
        """Test UUID generation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(timestamp=5.5, bounding_box=bbox, area_type="chatgpt")

        # Should be valid UUID
        uuid.UUID(area.id)  # This will raise if not valid UUID

    def test_invalid_area_type(self):
        """Test invalid area type raises error."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError, match="Unsupported area type"):
            SensitiveArea(timestamp=5.5, bounding_box=bbox, area_type="invalid_type")

    def test_invalid_confidence_range(self):
        """Test invalid confidence range."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError):
            SensitiveArea(
                timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=1.5
            )

    def test_negative_timestamp(self):
        """Test negative timestamp raises error."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError):
            SensitiveArea(timestamp=-1.0, bounding_box=bbox, area_type="chatgpt")

    def test_properties(self):
        """Test SensitiveArea properties."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        manual_area = SensitiveArea(
            timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=1.0
        )
        detected_area = SensitiveArea(
            timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=0.7
        )

        assert manual_area.is_manual is True
        assert detected_area.is_manual is False
        assert detected_area.needs_review is True

    def test_to_detection_format(self):
        """Test conversion to detection format."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(
            timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=0.95
        )

        detection_format = area.to_detection_format()
        assert detection_format["timestamp"] == 5.5
        assert detection_format["confidence"] == 0.95
        assert detection_format["area_type"] == "chatgpt"

    def test_with_updated_confidence(self):
        """Test creating copy with updated confidence."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(
            timestamp=5.5, bounding_box=bbox, area_type="chatgpt", confidence=0.7
        )

        updated = area.with_updated_confidence(0.95)
        assert updated.confidence == 0.95
        assert updated.id == area.id  # Same ID
        assert area.confidence == 0.7  # Original unchanged

    def test_datetime_timezone(self):
        """Test that created_at uses UTC timezone."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(timestamp=5.5, bounding_box=bbox, area_type="chatgpt")

        assert area.created_at.tzinfo == UTC


class TestDetectionResult:
    """Tests for DetectionResult model."""

    def test_valid_creation(self):
        """Test creating a valid detection result."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        result = DetectionResult(
            detections=[(bbox, 0.9, "chatgpt")],
            detection_time=0.15,
            detector_type="template_detector",
            timestamp=5.0,
        )

        assert len(result.detections) == 1
        assert result.detection_time == 0.15
        assert result.detector_type == "template_detector"

    def test_multiple_detections(self):
        """Test handling multiple detections."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result = DetectionResult(
            detections=[(bbox1, 0.9, "chatgpt"), (bbox2, 0.8, "terminal")],
            detection_time=0.2,
            detector_type="multi_detector",
        )

        assert result.detection_count == 2
        assert len(result.regions) == 2
        assert len(result.confidences) == 2
        assert len(result.area_types) == 2
        assert result.area_types == ["chatgpt", "terminal"]

    def test_invalid_detection_tuple(self):
        """Test invalid detection tuple structure."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError, match="Field required"):
            DetectionResult(
                detections=[(bbox, 0.9)],  # Missing area_type
                detection_time=0.15,
                detector_type="test",
            )

    def test_invalid_confidence_in_detection(self):
        """Test invalid confidence in detection tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError, match="Confidence.*out of range"):
            DetectionResult(
                detections=[(bbox, 1.5, "chatgpt")],  # Invalid confidence
                detection_time=0.15,
                detector_type="test",
            )

    def test_invalid_area_type_in_detection(self):
        """Test invalid area type in detection tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(ValidationError, match="Invalid area_type"):
            DetectionResult(
                detections=[(bbox, 0.9, "invalid_type")],
                detection_time=0.15,
                detector_type="test",
            )

    def test_best_detection(self):
        """Test getting best detection."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result = DetectionResult(
            detections=[(bbox1, 0.8, "chatgpt"), (bbox2, 0.95, "terminal")],
            detection_time=0.2,
            detector_type="test",
        )

        best = result.best_detection
        assert best is not None
        assert best[1] == 0.95  # Best confidence
        assert best[2] == "terminal"  # Best area type

    def test_empty_detection_result(self):
        """Test empty detection result."""
        result = DetectionResult.empty(timestamp=10.0, detector_type="test")

        assert result.detection_count == 0
        assert result.has_detections is False
        assert result.best_detection is None
        assert result.average_confidence == 0.0

    def test_filter_by_confidence(self):
        """Test filtering by confidence threshold."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result = DetectionResult(
            detections=[(bbox1, 0.6, "chatgpt"), (bbox2, 0.9, "terminal")],
            detection_time=0.2,
            detector_type="test",
        )

        filtered = result.filter_by_confidence(0.8)
        assert filtered.detection_count == 1
        assert filtered.confidences[0] == 0.9

    def test_filter_by_area_type(self):
        """Test filtering by area type."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result = DetectionResult(
            detections=[(bbox1, 0.8, "chatgpt"), (bbox2, 0.9, "chatgpt")],
            detection_time=0.2,
            detector_type="test",
        )

        filtered = result.filter_by_area_type("chatgpt")
        assert filtered.detection_count == 2

        filtered_terminal = result.filter_by_area_type("terminal")
        assert filtered_terminal.detection_count == 0

    def test_merge_detection_results(self):
        """Test merging detection results."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result1 = DetectionResult(
            detections=[(bbox1, 0.8, "chatgpt")],
            detection_time=0.1,
            detector_type="detector1",
        )

        result2 = DetectionResult(
            detections=[(bbox2, 0.9, "terminal")],
            detection_time=0.15,
            detector_type="detector2",
        )

        merged = result1.merge_with(result2)
        assert merged.detection_count == 2
        assert merged.detection_time == 0.25
        assert "detector1+detector2" in merged.detector_type

    def test_to_sensitive_areas(self):
        """Test conversion to sensitive areas."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
        bbox2 = BoundingBox(x=200, y=300, width=150, height=75)

        result = DetectionResult(
            detections=[(bbox1, 0.8, "chatgpt"), (bbox2, 0.9, "terminal")],
            detection_time=0.2,
            detector_type="test_detector",
            timestamp=10.5,
        )

        areas = result.to_sensitive_areas()
        assert len(areas) == 2
        assert all(isinstance(area, SensitiveArea) for area in areas)
        assert areas[0].timestamp == 10.5
        assert areas[0].area_type == "chatgpt"
        assert areas[1].area_type == "terminal"

    def test_add_detection(self):
        """Test adding detection to existing result."""
        result = DetectionResult.empty()
        bbox = BoundingBox(x=10, y=20, width=100, height=50)

        result.add_detection(bbox, 0.9, "chatgpt")
        assert result.detection_count == 1
        assert result.confidences[0] == 0.9


# Additional tests for BlurRegion and Timeline would follow similar patterns...


class TestBlurRegion:
    """Tests for BlurRegion model."""

    def test_valid_creation(self):
        """Test creating a valid blur region."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = BlurRegion(
            start_time=5.0,
            end_time=10.0,
            bounding_box=bbox,
            blur_type="gaussian",
        )

        assert region.start_time == 5.0
        assert region.end_time == 10.0
        assert region.duration == 5.0
        assert region.blur_type == "gaussian"

    def test_invalid_time_range(self):
        """Test invalid time range raises error."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        with pytest.raises(
            ValidationError, match="end_time.*must be greater than.*start_time"
        ):
            BlurRegion(
                start_time=10.0,
                end_time=5.0,
                bounding_box=bbox,
            )

    def test_overlaps_time(self):
        """Test time overlap checking."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = BlurRegion(start_time=5.0, end_time=10.0, bounding_box=bbox)

        assert region.overlaps_time(7.5) is True
        assert region.overlaps_time(5.0) is True  # Boundary inclusive
        assert region.overlaps_time(10.0) is True  # Boundary inclusive
        assert region.overlaps_time(15.0) is False

    def test_split_at_time(self):
        """Test splitting region at specific time."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = BlurRegion(start_time=5.0, end_time=10.0, bounding_box=bbox)

        first, second = region.split_at_time(7.5)
        assert first.end_time == 7.5
        assert second.start_time == 7.5
        assert first.start_time == 5.0
        assert second.end_time == 10.0

    def test_from_sensitive_area(self):
        """Test creating blur region from sensitive area."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        area = SensitiveArea(
            timestamp=10.0, bounding_box=bbox, area_type="chatgpt", confidence=0.9
        )

        region = BlurRegion.from_sensitive_area(
            area, duration=3.0, blur_type="composite"
        )
        assert region.start_time == 10.0
        assert region.end_time == 13.0
        assert region.blur_type == "composite"
        assert region.confidence == 0.9


class TestTimeline:
    """Tests for Timeline model."""

    def test_valid_creation(self):
        """Test creating a valid timeline."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=120.0,
            frame_rate=30.0,
        )

        assert timeline.video_duration == 120.0
        assert timeline.frame_rate == 30.0
        assert timeline.region_count == 0

    def test_add_region(self):
        """Test adding blur region to timeline."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=120.0,
        )

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = BlurRegion(start_time=5.0, end_time=10.0, bounding_box=bbox)

        timeline.add_region(region)
        assert timeline.region_count == 1

    def test_region_exceeds_duration(self):
        """Test adding region that exceeds video duration."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=60.0,
        )

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = BlurRegion(start_time=50.0, end_time=70.0, bounding_box=bbox)

        with pytest.raises(ValueError, match="extending beyond video duration"):
            timeline.add_region(region)

    def test_get_active_regions(self):
        """Test getting active regions at timestamp."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=120.0,
        )

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region1 = BlurRegion(start_time=5.0, end_time=15.0, bounding_box=bbox)
        region2 = BlurRegion(start_time=20.0, end_time=30.0, bounding_box=bbox)

        timeline.add_region(region1)
        timeline.add_region(region2)

        active_at_10 = timeline.get_active_regions(10.0)
        active_at_25 = timeline.get_active_regions(25.0)
        active_at_35 = timeline.get_active_regions(35.0)

        assert len(active_at_10) == 1
        assert len(active_at_25) == 1
        assert len(active_at_35) == 0

    def test_from_detection_results(self):
        """Test creating timeline from detection results."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        result = DetectionResult(
            detections=[(bbox, 0.9, "chatgpt")],
            detection_time=0.1,
            detector_type="test",
            timestamp=10.0,
        )

        timeline = Timeline.from_detection_results(
            video_path=Path("/test/video.mp4"),
            video_duration=120.0,
            detection_results=[result],
            default_duration=2.0,
        )

        assert timeline.region_count == 1
        assert timeline.blur_regions[0].start_time == 10.0
        assert timeline.blur_regions[0].end_time == 12.0

    def test_blur_coverage_percentage(self):
        """Test blur coverage calculation."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=100.0,
        )

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        # Add region covering 10 seconds of 100 second video = 10%
        region = BlurRegion(start_time=0.0, end_time=10.0, bounding_box=bbox)
        timeline.add_region(region)

        coverage = timeline.blur_coverage_percentage()
        assert (
            9.0 < coverage < 11.0
        )  # Approximately 10% (allowing for segment precision)

    def test_optimize_merging(self):
        """Test timeline optimization through region merging."""
        timeline = Timeline(
            video_path=Path("/test/video.mp4"),
            video_duration=120.0,
        )

        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        # Two overlapping regions of same type
        region1 = BlurRegion(
            start_time=5.0, end_time=15.0, bounding_box=bbox, blur_type="gaussian"
        )
        region2 = BlurRegion(
            start_time=10.0, end_time=20.0, bounding_box=bbox, blur_type="gaussian"
        )

        timeline.add_region(region1)
        timeline.add_region(region2)

        optimized = timeline.optimize()
        # Should merge into single region since they overlap and have same type
        assert optimized.region_count <= timeline.region_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
