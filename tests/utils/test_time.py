"""Tests for time utilities."""

from __future__ import annotations

import pytest

from mivideoeditor.utils.time import (
    MILLISECONDS_IN_SECOND,
    MINUTES_IN_HOUR,
    SECONDS_IN_MINUTE,
    TimeUtils,
)


class TestTimeUtils:
    """Test time utilities."""

    def test_parse_time_string_numeric(self):
        """Test parsing numeric time strings."""
        # Valid numeric times
        assert TimeUtils.parse_time_string("123.456") == 123.456
        assert TimeUtils.parse_time_string("0") == 0.0
        assert TimeUtils.parse_time_string("60") == 60.0
        assert TimeUtils.parse_time_string("3600.0") == 3600.0

    def test_parse_time_string_hms(self):
        """Test parsing HH:MM:SS time strings."""
        # Hours:Minutes:Seconds format
        assert TimeUtils.parse_time_string("01:30:45") == 5445.0  # 1*3600 + 30*60 + 45
        assert TimeUtils.parse_time_string("0:05:30") == 330.0  # 5*60 + 30
        assert TimeUtils.parse_time_string("12:00:00") == 43200.0  # 12*3600

        # Minutes:Seconds format
        assert TimeUtils.parse_time_string("05:30") == 330.0  # 5*60 + 30
        assert TimeUtils.parse_time_string("0:30") == 30.0  # 30 seconds

        # Seconds only
        assert TimeUtils.parse_time_string("45") == 45.0

    def test_parse_time_string_with_milliseconds(self):
        """Test parsing time strings with milliseconds."""
        assert TimeUtils.parse_time_string("01:30:45.123") == 5445.123
        assert TimeUtils.parse_time_string("05:30.5") == 330.5
        assert TimeUtils.parse_time_string("45.250") == 45.25
        assert TimeUtils.parse_time_string("1:23.45") == 83.45

    def test_parse_time_string_human_readable(self):
        """Test parsing human-readable time strings."""
        # Hours
        assert TimeUtils.parse_time_string("1h") == 3600.0
        assert TimeUtils.parse_time_string("2 hours") == 7200.0
        assert TimeUtils.parse_time_string("1.5h") == 5400.0

        # Minutes
        assert TimeUtils.parse_time_string("30m") == 1800.0
        assert TimeUtils.parse_time_string("45 minutes") == 2700.0
        assert TimeUtils.parse_time_string("2.5min") == 150.0

        # Seconds
        assert TimeUtils.parse_time_string("30s") == 30.0
        assert TimeUtils.parse_time_string("45 seconds") == 45.0
        assert TimeUtils.parse_time_string("1.5sec") == 1.5

        # Combined formats
        assert TimeUtils.parse_time_string("1h 30m 45s") == 5445.0
        assert TimeUtils.parse_time_string("2h 15m") == 8100.0
        assert TimeUtils.parse_time_string("45m 30s") == 2730.0

    def test_parse_time_string_edge_cases(self):
        """Test edge cases in time parsing."""
        # Leading/trailing whitespace
        assert TimeUtils.parse_time_string("  123.45  ") == 123.45
        assert TimeUtils.parse_time_string("  1h 30m  ") == 5400.0

        # Case insensitive
        assert TimeUtils.parse_time_string("1H 30M 45S") == 5445.0
        assert TimeUtils.parse_time_string("1HOUR 30MIN") == 5400.0

    def test_parse_time_string_validation(self):
        """Test validation in time parsing."""
        # Invalid minute/second values
        with pytest.raises(ValueError, match="Invalid time components"):
            TimeUtils.parse_time_string("01:60:30")  # 60 minutes

        with pytest.raises(ValueError, match="Invalid time components"):
            TimeUtils.parse_time_string("01:30:60")  # 60 seconds

        with pytest.raises(ValueError, match="Invalid time components"):
            TimeUtils.parse_time_string("01:70:30")  # 70 minutes

    def test_parse_time_string_errors(self):
        """Test error cases in time parsing."""
        # Empty string
        with pytest.raises(ValueError, match="Empty time string"):
            TimeUtils.parse_time_string("")

        with pytest.raises(ValueError, match="Empty time string"):
            TimeUtils.parse_time_string("   ")

        # Unrecognized format
        with pytest.raises(ValueError, match="Unrecognized time format"):
            TimeUtils.parse_time_string("invalid")

        with pytest.raises(ValueError, match="Unrecognized time format"):
            TimeUtils.parse_time_string("1d 2h")  # Days not supported

        # No time components found
        with pytest.raises(ValueError, match="No time components found"):
            TimeUtils.parse_time_string("hello world")

    def test_format_duration_hms(self):
        """Test HMS duration formatting."""
        # Basic HH:MM:SS format
        assert TimeUtils.format_duration(3661, "hms", 0) == "01:01:01"
        assert TimeUtils.format_duration(3600, "hms", 0) == "01:00:00"
        assert TimeUtils.format_duration(61, "hms", 0) == "01:01"
        assert TimeUtils.format_duration(45, "hms", 0) == "00:45"

        # With milliseconds
        assert TimeUtils.format_duration(3661.123, "hms", 1) == "01:01:01.1"
        assert TimeUtils.format_duration(3661.123, "hms", 3) == "01:01:01.123"
        assert TimeUtils.format_duration(45.5, "hms", 1) == "00:45.5"

    def test_format_duration_compact(self):
        """Test compact duration formatting."""
        assert TimeUtils.format_duration(3661, "compact", 0) == "1h1m1s"
        assert TimeUtils.format_duration(3600, "compact", 0) == "1h"
        assert TimeUtils.format_duration(61, "compact", 0) == "1m1s"
        assert TimeUtils.format_duration(45, "compact", 0) == "45s"
        assert TimeUtils.format_duration(0, "compact", 0) == "0s"

        # With decimals
        assert TimeUtils.format_duration(3661.5, "compact", 1) == "1h1m1.5s"
        assert TimeUtils.format_duration(1800.25, "compact", 2) == "30m0.25s"

    def test_format_duration_long(self):
        """Test long form duration formatting."""
        assert (
            TimeUtils.format_duration(3661, "long", 0)
            == "1 hour, 1 minute, and 1 second"
        )
        assert TimeUtils.format_duration(3600, "long", 0) == "1 hour"
        assert TimeUtils.format_duration(7200, "long", 0) == "2 hours"
        assert TimeUtils.format_duration(61, "long", 0) == "1 minute and 1 second"
        assert TimeUtils.format_duration(120, "long", 0) == "2 minutes"
        assert TimeUtils.format_duration(1, "long", 0) == "1 second"
        assert TimeUtils.format_duration(2, "long", 0) == "2 seconds"
        assert TimeUtils.format_duration(0, "long", 0) == "0 seconds"

    def test_format_duration_minimal(self):
        """Test minimal duration formatting."""
        assert TimeUtils.format_duration(7323, "minimal", 0) == "2h2m3s"  # 2:02:03
        assert TimeUtils.format_duration(3600, "minimal", 0) == "1h"
        assert TimeUtils.format_duration(1800, "minimal", 0) == "30m"
        assert TimeUtils.format_duration(45, "minimal", 0) == "45s"
        assert TimeUtils.format_duration(90, "minimal", 0) == "1m30s"

    def test_format_duration_negative(self):
        """Test formatting negative durations."""
        assert TimeUtils.format_duration(-3661, "hms", 0) == "-01:01:01"
        assert TimeUtils.format_duration(-45, "compact", 0) == "-45s"
        assert (
            TimeUtils.format_duration(-3661, "long", 0)
            == "-1 hour, 1 minute, and 1 second"
        )
        assert TimeUtils.format_duration(-45, "minimal", 0) == "-45s"

    def test_format_duration_invalid_format(self):
        """Test invalid format type."""
        with pytest.raises(ValueError, match="Unknown format_type"):
            TimeUtils.format_duration(123, "invalid", 0)

    def test_merge_overlapping_ranges_basic(self):
        """Test basic range merging."""
        # Non-overlapping ranges
        ranges = [(0, 5), (10, 15), (20, 25)]
        result = TimeUtils.merge_overlapping_ranges(ranges)
        assert result == [(0, 5), (10, 15), (20, 25)]

        # Overlapping ranges
        ranges = [(0, 5), (3, 8), (7, 12)]
        result = TimeUtils.merge_overlapping_ranges(ranges)
        assert result == [(0, 12)]

        # Adjacent ranges
        ranges = [(0, 5), (5, 10), (10, 15)]
        result = TimeUtils.merge_overlapping_ranges(ranges)
        assert result == [(0, 15)]

    def test_merge_overlapping_ranges_with_tolerance(self):
        """Test range merging with tolerance."""
        # Ranges within tolerance
        ranges = [(0, 5), (5.5, 10)]
        result = TimeUtils.merge_overlapping_ranges(ranges, tolerance=1.0)
        assert result == [(0, 10)]

        # Ranges outside tolerance
        ranges = [(0, 5), (7, 10)]
        result = TimeUtils.merge_overlapping_ranges(ranges, tolerance=1.0)
        assert result == [(0, 5), (7, 10)]

    def test_merge_overlapping_ranges_edge_cases(self):
        """Test edge cases in range merging."""
        # Empty list
        assert TimeUtils.merge_overlapping_ranges([]) == []

        # Invalid ranges (start >= end)
        ranges = [(5, 3), (0, 5), (10, 10)]
        result = TimeUtils.merge_overlapping_ranges(ranges)
        assert result == [(0, 5)]

        # Single range
        ranges = [(0, 5)]
        result = TimeUtils.merge_overlapping_ranges(ranges)
        assert result == [(0, 5)]

    def test_find_gaps_in_ranges(self):
        """Test finding gaps between ranges."""
        # Simple gaps
        ranges = [(1, 3), (5, 7), (10, 12)]
        gaps = TimeUtils.find_gaps_in_ranges(ranges, total_duration=15)
        expected = [(0.0, 1), (3, 5), (7, 10), (12, 15)]
        assert gaps == expected

        # No gaps (continuous coverage)
        ranges = [(0, 5), (5, 10), (10, 15)]
        gaps = TimeUtils.find_gaps_in_ranges(ranges, total_duration=15)
        assert gaps == []

    def test_find_gaps_in_ranges_with_min_gap_size(self):
        """Test gap finding with minimum gap size."""
        ranges = [(0, 5), (5.05, 10)]  # Very small gap
        gaps = TimeUtils.find_gaps_in_ranges(
            ranges, total_duration=15, min_gap_size=0.1
        )
        assert len(gaps) == 1  # Gap too small, ignored
        assert gaps == [(10, 15)]

        # Larger gap that meets minimum
        ranges = [(0, 5), (6, 10)]
        gaps = TimeUtils.find_gaps_in_ranges(
            ranges, total_duration=15, min_gap_size=0.5
        )
        assert (5, 6) in gaps

    def test_find_gaps_edge_cases(self):
        """Test edge cases in gap finding."""
        # Zero duration
        gaps = TimeUtils.find_gaps_in_ranges([(1, 3)], total_duration=0)
        assert gaps == []

        # Empty ranges
        gaps = TimeUtils.find_gaps_in_ranges([], total_duration=10)
        assert gaps == [(0.0, 10)]

        # Ranges beyond total duration
        ranges = [(0, 5), (15, 20)]  # Second range beyond duration
        gaps = TimeUtils.find_gaps_in_ranges(ranges, total_duration=10)
        assert gaps == [(5, 10)]

    def test_calculate_range_coverage(self):
        """Test range coverage calculation."""
        ranges = [(0, 5), (10, 15)]
        coverage = TimeUtils.calculate_range_coverage(ranges, total_duration=20)

        assert coverage["covered_duration"] == 10.0
        assert coverage["coverage_percentage"] == 50.0
        assert coverage["gap_duration"] == 10.0
        assert coverage["gap_percentage"] == 50.0
        assert coverage["range_count"] == 2
        assert coverage["merged_range_count"] == 2

    def test_calculate_range_coverage_overlapping(self):
        """Test coverage calculation with overlapping ranges."""
        ranges = [(0, 10), (5, 15), (12, 18)]
        coverage = TimeUtils.calculate_range_coverage(ranges, total_duration=20)

        # Should merge to (0, 18)
        assert coverage["covered_duration"] == 18.0
        assert coverage["coverage_percentage"] == 90.0
        assert coverage["range_count"] == 3
        assert coverage["merged_range_count"] == 1

    def test_calculate_range_coverage_edge_cases(self):
        """Test coverage calculation edge cases."""
        # Zero duration
        coverage = TimeUtils.calculate_range_coverage([(1, 3)], total_duration=0)
        assert coverage["coverage_percentage"] == 0.0
        assert coverage["merged_range_count"] == 0

        # Ranges beyond total duration
        ranges = [(0, 5), (15, 25)]  # Extends beyond duration=10
        coverage = TimeUtils.calculate_range_coverage(ranges, total_duration=10)
        assert coverage["covered_duration"] == 5.0  # Only first range counts

    def test_split_duration_into_chunks(self):
        """Test splitting duration into chunks."""
        # Basic chunking without overlap
        chunks = TimeUtils.split_duration_into_chunks(total_duration=10, chunk_size=3)
        expected = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 10.0)]
        assert chunks == expected

        # With overlap
        chunks = TimeUtils.split_duration_into_chunks(
            total_duration=10, chunk_size=4, overlap=1
        )
        expected = [(0.0, 4.0), (3.0, 7.0), (6.0, 10.0)]
        assert chunks == expected

    def test_split_duration_into_chunks_edge_cases(self):
        """Test edge cases in duration splitting."""
        # Zero duration
        chunks = TimeUtils.split_duration_into_chunks(total_duration=0, chunk_size=5)
        assert chunks == []

        # Zero chunk size
        chunks = TimeUtils.split_duration_into_chunks(total_duration=10, chunk_size=0)
        assert chunks == []

        # Overlap >= chunk size
        with pytest.raises(ValueError, match="Overlap .* must be less than chunk_size"):
            TimeUtils.split_duration_into_chunks(
                total_duration=10, chunk_size=5, overlap=5
            )

    def test_validate_time_ranges_valid(self):
        """Test validation of valid time ranges."""
        ranges = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]
        result = TimeUtils.validate_time_ranges(ranges, total_duration=30.0)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_time_ranges_errors(self):
        """Test validation with various error conditions."""
        # Invalid range format
        ranges = [(0, 5), "invalid", (10, 15)]
        result = TimeUtils.validate_time_ranges(ranges)
        assert not result.is_valid
        assert any("must be a (start, end) tuple" in error for error in result.errors)

        # Non-numeric values
        ranges = [(0, 5), ("a", "b")]
        result = TimeUtils.validate_time_ranges(ranges)
        assert not result.is_valid
        assert any("non-numeric values" in error for error in result.errors)

        # Negative times
        ranges = [(-1, 5), (0, -3)]
        result = TimeUtils.validate_time_ranges(ranges)
        assert not result.is_valid
        assert any("negative start time" in error for error in result.errors)
        assert any("negative end time" in error for error in result.errors)

        # Start >= end
        ranges = [(5, 3), (10, 10)]
        result = TimeUtils.validate_time_ranges(ranges)
        assert not result.is_valid
        assert any("start >= end" in error for error in result.errors)

    def test_validate_time_ranges_warnings(self):
        """Test validation warnings."""
        # Very short ranges
        ranges = [(0.0, 0.005), (10.0, 15.0)]
        result = TimeUtils.validate_time_ranges(ranges)
        assert result.is_valid is True
        assert any("very short" in warning.lower() for warning in result.warnings)

        # Ranges beyond total duration
        ranges = [(0.0, 5.0), (25.0, 30.0)]
        result = TimeUtils.validate_time_ranges(ranges, total_duration=20.0)
        assert result.is_valid is True
        assert any("after total duration" in warning for warning in result.warnings)

        # Overlapping ranges
        ranges = [(0.0, 10.0), (5.0, 15.0)]
        result = TimeUtils.validate_time_ranges(ranges)
        assert result.is_valid is True
        assert any("overlapping ranges" in warning for warning in result.warnings)

    def test_validate_time_ranges_coverage_analysis(self):
        """Test coverage analysis in validation."""
        # High coverage
        ranges = [(0.0, 95.0)]
        result = TimeUtils.validate_time_ranges(ranges, total_duration=100.0)
        assert any("High coverage" in warning for warning in result.warnings)

        # Low coverage
        ranges = [(0.0, 5.0)]
        result = TimeUtils.validate_time_ranges(ranges, total_duration=100.0)
        assert any("Low coverage" in warning for warning in result.warnings)

    def test_timestamp_to_frame_number(self):
        """Test timestamp to frame number conversion."""
        # Standard frame rates
        assert TimeUtils.timestamp_to_frame_number(1.0, 30.0, "nearest") == 30
        assert TimeUtils.timestamp_to_frame_number(1.5, 24.0, "nearest") == 36
        assert TimeUtils.timestamp_to_frame_number(2.0, 60.0, "floor") == 120
        assert TimeUtils.timestamp_to_frame_number(1.9, 30.0, "ceil") == 57

        # Edge cases
        assert TimeUtils.timestamp_to_frame_number(0.0, 30.0) == 0
        assert TimeUtils.timestamp_to_frame_number(0.033, 30.0, "nearest") == 1

    def test_timestamp_to_frame_number_errors(self):
        """Test timestamp conversion error cases."""
        # Invalid frame rate
        with pytest.raises(ValueError, match="Frame rate must be positive"):
            TimeUtils.timestamp_to_frame_number(1.0, 0.0)

        with pytest.raises(ValueError, match="Frame rate must be positive"):
            TimeUtils.timestamp_to_frame_number(1.0, -30.0)

        # Invalid rounding method
        with pytest.raises(ValueError, match="Unknown rounding method"):
            TimeUtils.timestamp_to_frame_number(1.0, 30.0, "invalid")

    def test_frame_number_to_timestamp(self):
        """Test frame number to timestamp conversion."""
        assert TimeUtils.frame_number_to_timestamp(30, 30.0) == 1.0
        assert TimeUtils.frame_number_to_timestamp(60, 24.0) == 2.5
        assert TimeUtils.frame_number_to_timestamp(0, 30.0) == 0.0
        assert TimeUtils.frame_number_to_timestamp(90, 60.0) == 1.5

    def test_frame_number_to_timestamp_errors(self):
        """Test frame to timestamp conversion errors."""
        with pytest.raises(ValueError, match="Frame rate must be positive"):
            TimeUtils.frame_number_to_timestamp(30, 0.0)

    def test_create_timestamp_sequence(self):
        """Test timestamp sequence creation."""
        # Basic sequence
        seq = TimeUtils.create_timestamp_sequence(0.0, 5.0, 1.0)
        assert seq == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        # Non-integer steps
        seq = TimeUtils.create_timestamp_sequence(0.0, 2.5, 0.5)
        assert seq == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

        # Empty sequence (end <= start)
        seq = TimeUtils.create_timestamp_sequence(5.0, 3.0, 1.0)
        assert seq == []

        seq = TimeUtils.create_timestamp_sequence(5.0, 5.0, 1.0)
        assert seq == []

    def test_create_timestamp_sequence_errors(self):
        """Test timestamp sequence creation errors."""
        with pytest.raises(ValueError, match="Step must be positive"):
            TimeUtils.create_timestamp_sequence(0.0, 5.0, 0.0)

        with pytest.raises(ValueError, match="Step must be positive"):
            TimeUtils.create_timestamp_sequence(0.0, 5.0, -1.0)

    def test_get_current_iso_timestamp(self):
        """Test current ISO timestamp generation."""
        timestamp = TimeUtils.get_current_iso_timestamp()

        # Should be a valid ISO format string
        assert isinstance(timestamp, str)
        assert "T" in timestamp
        assert timestamp.endswith("+00:00") or timestamp.endswith("Z")

    def test_parse_iso_timestamp(self):
        """Test ISO timestamp parsing."""
        # Valid ISO timestamps
        iso_str = "2023-12-25T15:30:45+00:00"
        dt = TimeUtils.parse_iso_timestamp(iso_str)
        assert dt.year == 2023
        assert dt.month == 12
        assert dt.day == 25
        assert dt.hour == 15
        assert dt.minute == 30
        assert dt.second == 45

        # With microseconds
        iso_str = "2023-12-25T15:30:45.123456+00:00"
        dt = TimeUtils.parse_iso_timestamp(iso_str)
        assert dt.microsecond == 123456

    def test_parse_iso_timestamp_errors(self):
        """Test ISO timestamp parsing errors."""
        with pytest.raises(ValueError, match="Invalid ISO timestamp"):
            TimeUtils.parse_iso_timestamp("invalid timestamp")

        with pytest.raises(ValueError, match="Invalid ISO timestamp"):
            TimeUtils.parse_iso_timestamp("2023-13-45T25:70:90")

    def test_constants(self):
        """Test time constants are correct."""
        assert MINUTES_IN_HOUR == 60
        assert SECONDS_IN_MINUTE == 60
        assert MILLISECONDS_IN_SECOND == 1000

    def test_round_trip_conversions(self):
        """Test round-trip conversions maintain accuracy."""
        # Time string parsing and formatting
        original = "1h 30m 45s"
        parsed = TimeUtils.parse_time_string(original)
        formatted = TimeUtils.format_duration(parsed, "compact", 0)
        assert formatted == "1h30m45s"

        # Frame/timestamp conversions
        original_timestamp = 1.5
        frame_rate = 30.0
        frame_num = TimeUtils.timestamp_to_frame_number(original_timestamp, frame_rate)
        converted_back = TimeUtils.frame_number_to_timestamp(frame_num, frame_rate)
        assert (
            abs(converted_back - original_timestamp) < 0.1
        )  # Allow small rounding error
