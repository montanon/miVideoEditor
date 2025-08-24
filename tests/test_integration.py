"""Integration tests for the complete Phase 1 pipeline."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mivideoeditor.core import (
    BlurRegion,
    BoundingBox,
    DetectionResult,
    SensitiveArea,
    Timeline,
    ValidationResult,
)


def test_complete_phase1_pipeline() -> None:
    """Test the complete Phase 1 data flow pipeline."""
    # 1. Create bounding boxes for different sensitive areas
    chatgpt_bbox = BoundingBox(x=100, y=100, width=600, height=400)
    terminal_bbox = BoundingBox(x=800, y=500, width=400, height=300)

    # 2. Create detection results with multiple areas per frame
    frame1_result = DetectionResult(
        detections=[
            (chatgpt_bbox, 0.95, "chatgpt"),
            (terminal_bbox, 0.88, "terminal"),
        ],
        detection_time=0.125,
        detector_type="template_detector",
        timestamp=10.0,
    )

    frame2_result = DetectionResult(
        detections=[(chatgpt_bbox, 0.92, "chatgpt")],
        detection_time=0.110,
        detector_type="template_detector",
        timestamp=25.0,
    )

    # 3. Create timeline from detection results
    timeline = Timeline.from_detection_results(
        video_path=Path("/test/screen_recording.mp4"),
        video_duration=300.0,  # 5 minutes
        detection_results=[frame1_result, frame2_result],
        default_duration=3.0,  # 3 second blur regions
    )

    # 4. Verify the pipeline worked correctly
    assert timeline.region_count == 3  # 2 areas from frame1 + 1 from frame2
    assert timeline.total_blur_duration() == 9.0  # 3 regions x 3 seconds each

    # 5. Test temporal queries
    active_at_11s = timeline.get_active_regions(11.0)
    assert len(active_at_11s) == 2  # Both areas from frame1 active

    active_at_26s = timeline.get_active_regions(26.0)
    assert len(active_at_26s) == 1  # Only chatgpt from frame2 active

    active_at_50s = timeline.get_active_regions(50.0)
    assert len(active_at_50s) == 0  # No regions active

    # 6. Test optimization
    optimized = timeline.optimize()
    # Should have same or fewer regions due to potential merging
    assert optimized.region_count <= timeline.region_count

    # 7. Test validation
    validation_issues = timeline.validate()
    assert isinstance(validation_issues, list)

    # 8. Test individual model conversions
    sensitive_areas = []
    for result in [frame1_result, frame2_result]:
        sensitive_areas.extend(result.to_sensitive_areas())

    assert len(sensitive_areas) == 3
    assert all(isinstance(area, SensitiveArea) for area in sensitive_areas)

    # 9. Test ValidationResult usage
    validation = ValidationResult.success()
    for area in sensitive_areas:
        if area.confidence < 0.9:
            validation.add_warning(f"Low confidence area: {area.id}")

    assert validation.is_valid
    assert validation.has_warnings

    # 10. Test blur region operations
    blur_region = BlurRegion.from_sensitive_area(
        sensitive_areas[0], duration=5.0, blur_type="composite"
    )

    # Test splitting
    first_half, second_half = blur_region.split_at_time(
        blur_region.start_time + blur_region.duration / 2
    )

    assert first_half.duration + second_half.duration == blur_region.duration
    assert first_half.end_time == second_half.start_time


def test_error_handling_pipeline() -> None:
    """Test error handling across the pipeline."""
    # Test invalid bounding box
    with pytest.raises(ValidationError):
        BoundingBox(x=-1, y=0, width=100, height=100)

    # Test invalid area type
    bbox = BoundingBox(x=0, y=0, width=100, height=100)
    with pytest.raises(ValidationError):
        SensitiveArea(timestamp=0.0, bounding_box=bbox, area_type="invalid")

    # Test invalid time range for blur region
    with pytest.raises(ValidationError):
        BlurRegion(
            start_time=10.0,
            end_time=5.0,
            bounding_box=bbox,  # Invalid time range
        )

    # Test timeline validation
    timeline = Timeline(video_path=Path("/test/video.mp4"), video_duration=60.0)

    # Try to add region that exceeds duration
    invalid_region = BlurRegion(start_time=50.0, end_time=70.0, bounding_box=bbox)

    with pytest.raises(ValueError, match="extending beyond video duration"):
        timeline.add_region(invalid_region)


if __name__ == "__main__":
    test_complete_phase1_pipeline()
    test_error_handling_pipeline()
