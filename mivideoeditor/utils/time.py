"""Time utilities for parsing, formatting, and manipulating timestamps."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from mivideoeditor.core.models import ValidationResult

MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
MILLISECONDS_IN_SECOND = 1000


class TimeUtils:
    """Utilities for time parsing, formatting, and manipulation."""

    @staticmethod
    def parse_time_string(time_str: str) -> float:
        """Parse various time string formats to seconds."""
        time_str = time_str.strip()
        if not time_str:
            msg = "Empty time string"
            raise ValueError(msg)

        # Try numeric format first (simple seconds)
        try:
            return float(time_str)
        except ValueError:
            pass

        # Try HH:MM:SS.mmm format and variants
        colon_match = re.match(
            r"^(?:(\d{1,2}):)?(?:(\d{1,2}):)?(\d{1,2})(?:\.(\d{1,3}))?$", time_str
        )
        if colon_match:
            hours_str, minutes_str, seconds_str, millis_str = colon_match.groups()

            hours = int(hours_str) if hours_str else 0
            minutes = int(minutes_str) if minutes_str else 0
            seconds = int(seconds_str)
            millis = int(millis_str.ljust(3, "0")) if millis_str else 0

            # Validation
            if minutes >= MINUTES_IN_HOUR or seconds >= SECONDS_IN_MINUTE:
                msg = f"Invalid time components in: {time_str}"
                raise ValueError(msg)

            return (
                hours * SECONDS_IN_MINUTE * MINUTES_IN_HOUR
                + minutes * SECONDS_IN_MINUTE
                + seconds
                + millis / MILLISECONDS_IN_SECOND
            )

        # Try human-readable formats
        try:
            return TimeUtils._parse_human_readable(time_str)
        except ValueError:
            pass

        msg = f"Unrecognized time format: {time_str}"
        raise ValueError(msg)

    @staticmethod
    def _parse_human_readable(time_str: str) -> float:
        """Parse human-readable time strings like '1h 30m 45s'."""
        time_str = time_str.lower().replace(",", " ")

        # Define patterns for different time units
        patterns = [
            (
                r"(\d+(?:\.\d+)?)\s*h(?:our)?s?",
                SECONDS_IN_MINUTE * MINUTES_IN_HOUR,
            ),  # hours
            (r"(\d+(?:\.\d+)?)\s*m(?:in|inute)?s?", SECONDS_IN_MINUTE),  # minutes
            (r"(\d+(?:\.\d+)?)\s*s(?:ec|econd)?s?", 1),  # seconds
        ]

        total_seconds = 0.0
        found_any = False

        for pattern, multiplier in patterns:
            matches = re.findall(pattern, time_str)
            for match in matches:
                total_seconds += float(match) * multiplier
                found_any = True

        if not found_any:
            msg = f"No time components found in: {time_str}"
            raise ValueError(msg)

        return total_seconds

    @staticmethod
    def format_duration(
        seconds: float, format_type: str = "hms", precision: int = 1
    ) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 0:
            return f"-{TimeUtils.format_duration(-seconds, format_type, precision)}"

        if format_type == "hms":
            return TimeUtils._format_hms(seconds, precision)
        if format_type == "compact":
            return TimeUtils._format_compact(seconds, precision)
        if format_type == "long":
            return TimeUtils._format_long(seconds, precision)
        if format_type == "minimal":
            return TimeUtils._format_minimal(seconds, precision)
        msg = f"Unknown format_type: {format_type}"
        raise ValueError(msg)

    @staticmethod
    def _format_hms(seconds: float, precision: int) -> str:
        """Format as HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            if precision > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:0{precision + 3}.{precision}f}"
            return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"
        if precision > 0:
            return f"{minutes:02d}:{secs:0{precision + 3}.{precision}f}"
        return f"{minutes:02d}:{int(secs):02d}"

    @staticmethod
    def _format_compact(seconds: float, precision: int) -> str:
        """Format as compact notation: 1h30m45s."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if nothing else
            if precision > 0 and secs != int(secs):
                parts.append(f"{secs:.{precision}f}s")
            else:
                parts.append(f"{int(secs)}s")

        return "".join(parts)

    @staticmethod
    def _format_long(seconds: float, precision: int) -> str:
        """Format as long form: 1 hour, 30 minutes, 45 seconds."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            if precision > 0 and secs != int(secs):
                parts.append(f"{secs:.{precision}f} second{'s' if secs != 1 else ''}")
            else:
                sec_int = int(secs)
                parts.append(f"{sec_int} second{'s' if sec_int != 1 else ''}")

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    @staticmethod
    def _format_minimal(seconds: float, precision: int) -> str:
        """Format in minimal form (shortest representation)."""
        if seconds < SECONDS_IN_MINUTE:
            if precision > 0 and seconds != int(seconds):
                return f"{seconds:.{precision}f}s"
            return f"{int(seconds)}s"
        if seconds < SECONDS_IN_MINUTE * MINUTES_IN_HOUR:
            minutes = int(seconds // SECONDS_IN_MINUTE)
            remaining_secs = seconds % SECONDS_IN_MINUTE
            if remaining_secs == 0:
                return f"{minutes}m"
            return f"{minutes}m{int(remaining_secs)}s"
        hours = int(seconds // SECONDS_IN_MINUTE * MINUTES_IN_HOUR)
        remaining_minutes = int(
            (seconds % SECONDS_IN_MINUTE * MINUTES_IN_HOUR) // SECONDS_IN_MINUTE
        )
        remaining_secs = seconds % SECONDS_IN_MINUTE

        if remaining_minutes == 0 and remaining_secs == 0:
            return f"{hours}h"
        if remaining_secs == 0:
            return f"{hours}h{remaining_minutes}m"
        return f"{hours}h{remaining_minutes}m{int(remaining_secs)}s"

    @staticmethod
    def merge_overlapping_ranges(
        ranges: list[tuple[float, float]], tolerance: float = 0.0
    ) -> list[tuple[float, float]]:
        """Merge overlapping or adjacent time ranges."""
        if not ranges:
            return []

        # Validate and sort ranges
        valid_ranges = []
        for start, end in ranges:
            if start >= end:
                continue  # Skip invalid ranges
            valid_ranges.append((start, end))

        if not valid_ranges:
            return []

        # Sort by start time
        sorted_ranges = sorted(valid_ranges)

        merged = [sorted_ranges[0]]

        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]

            # Check if ranges overlap or are within tolerance
            if current_start <= last_end + tolerance:
                # Merge ranges
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new range
                merged.append((current_start, current_end))

        return merged

    @staticmethod
    def find_gaps_in_ranges(
        ranges: list[tuple[float, float]],
        total_duration: float,
        min_gap_size: float = 0.1,
    ) -> list[tuple[float, float]]:
        """Find gaps between time ranges within a total duration."""
        if total_duration <= 0:
            return []

        # Merge overlapping ranges first
        merged_ranges = TimeUtils.merge_overlapping_ranges(ranges)

        if not merged_ranges:
            return [(0.0, total_duration)]

        gaps = []
        current_pos = 0.0

        for start, end in merged_ranges:
            # Check for gap before this range
            if start > current_pos + min_gap_size:
                gaps.append((current_pos, start))

            current_pos = max(current_pos, end)

        # Check for gap after last range
        if current_pos < total_duration - min_gap_size:
            gaps.append((current_pos, total_duration))

        return gaps

    @staticmethod
    def calculate_range_coverage(
        ranges: list[tuple[float, float]], total_duration: float
    ) -> dict[str, float]:
        """Calculate coverage statistics for time ranges."""
        if total_duration <= 0:
            return {
                "covered_duration": 0.0,
                "coverage_percentage": 0.0,
                "gap_duration": 0.0,
                "gap_percentage": 0.0,
                "range_count": len(ranges),
                "merged_range_count": 0,
            }

        # Merge overlapping ranges
        merged_ranges = TimeUtils.merge_overlapping_ranges(ranges)

        # Calculate covered duration
        covered_duration = 0.0
        for start, end in merged_ranges:
            # Clamp to total duration
            clamped_start = max(0.0, min(start, total_duration))
            clamped_end = max(0.0, min(end, total_duration))
            if clamped_end > clamped_start:
                covered_duration += clamped_end - clamped_start

        coverage_percentage = (covered_duration / total_duration) * 100.0
        gap_duration = total_duration - covered_duration
        gap_percentage = 100.0 - coverage_percentage

        return {
            "covered_duration": covered_duration,
            "coverage_percentage": coverage_percentage,
            "gap_duration": gap_duration,
            "gap_percentage": gap_percentage,
            "range_count": len(ranges),
            "merged_range_count": len(merged_ranges),
        }

    @staticmethod
    def split_duration_into_chunks(
        total_duration: float, chunk_size: float, overlap: float = 0.0
    ) -> list[tuple[float, float]]:
        """Split a duration into overlapping chunks."""
        if total_duration <= 0 or chunk_size <= 0:
            return []

        if overlap >= chunk_size:
            msg = f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)

        chunks = []
        step = chunk_size - overlap
        current_start = 0.0

        while current_start < total_duration:
            current_end = min(current_start + chunk_size, total_duration)

            # Only add chunk if it has meaningful duration
            if current_end > current_start:
                chunks.append((current_start, current_end))

            current_start += step

            # Prevent infinite loop with very small steps
            if step <= 0:
                break

        return chunks

    @staticmethod
    def validate_time_ranges(
        ranges: list[tuple[float, float]], total_duration: float | None = None
    ) -> ValidationResult:
        """Validate a list of time ranges for consistency and bounds."""
        result = ValidationResult(is_valid=True)

        if len(ranges) == 0:
            result.add_warning("Empty ranges list")
            return result

        valid_ranges = []

        for i, range_tuple in enumerate(ranges):
            if not isinstance(range_tuple, (tuple, list)) or len(range_tuple) != 2:
                result.add_error(f"Range {i} must be a (start, end) tuple")
                continue

            try:
                start, end = float(range_tuple[0]), float(range_tuple[1])
            except (ValueError, TypeError):
                result.add_error(
                    f"Range {i} contains non-numeric values: {range_tuple}"
                )
                continue

            # Basic range validation
            if start < 0:
                result.add_error(f"Range {i} has negative start time: {start}")

            if end < 0:
                result.add_error(f"Range {i} has negative end time: {end}")

            if start >= end:
                result.add_error(f"Range {i} has start >= end: {start} >= {end}")
                continue

            # Duration bounds checking
            if total_duration is not None:
                if start > total_duration:
                    result.add_warning(
                        f"Range {i} starts after total duration: {start} > {total_duration}"
                    )

                if end > total_duration:
                    result.add_warning(
                        f"Range {i} ends after total duration: {end} > {total_duration}"
                    )

            # Very short ranges
            duration = end - start
            if duration < 0.01:  # Less than 10ms
                result.add_warning(f"Range {i} is very short: {duration:.3f}s")

            valid_ranges.append((start, end))

        if not valid_ranges:
            result.add_error("No valid ranges found")
            return result

        # Check for overlaps
        merged_ranges = TimeUtils.merge_overlapping_ranges(valid_ranges)
        if len(merged_ranges) < len(valid_ranges):
            overlap_count = len(valid_ranges) - len(merged_ranges)
            result.add_warning(f"Found {overlap_count} overlapping ranges")

        # Coverage analysis
        if total_duration is not None:
            coverage_stats = TimeUtils.calculate_range_coverage(
                valid_ranges, total_duration
            )
            coverage_pct = coverage_stats["coverage_percentage"]

            result.context.update(coverage_stats)

            if coverage_pct > 90:
                result.add_warning(
                    f"High coverage: {coverage_pct:.1f}% of total duration"
                )
            elif coverage_pct < 10:
                result.add_warning(
                    f"Low coverage: {coverage_pct:.1f}% of total duration"
                )

        return result

    @staticmethod
    def timestamp_to_frame_number(
        timestamp: float, frame_rate: float, rounding: str = "nearest"
    ) -> int:
        """Convert timestamp to frame number."""
        if frame_rate <= 0:
            msg = f"Frame rate must be positive, got: {frame_rate}"
            raise ValueError(msg)

        frame_float = timestamp * frame_rate

        if rounding == "nearest":
            return round(frame_float)
        if rounding == "floor":
            return int(frame_float)
        if rounding == "ceil":
            return math.ceil(frame_float)

        msg = f"Unknown rounding method: {rounding}"
        raise ValueError(msg)

    @staticmethod
    def frame_number_to_timestamp(frame_number: int, frame_rate: float) -> float:
        """Convert frame number to timestamp."""
        if frame_rate <= 0:
            msg = f"Frame rate must be positive, got: {frame_rate}"
            raise ValueError(msg)

        return frame_number / frame_rate

    @staticmethod
    def create_timestamp_sequence(
        start_time: float, end_time: float, step: float
    ) -> list[float]:
        """Create a sequence of timestamps with regular intervals."""
        if step <= 0:
            msg = f"Step must be positive, got: {step}"
            raise ValueError(msg)

        if end_time <= start_time:
            return []

        timestamps = []
        current = start_time

        while current <= end_time:
            timestamps.append(current)
            current += step

        return timestamps

    @staticmethod
    def get_current_iso_timestamp() -> str:
        """Get current timestamp in ISO 8601 format with timezone."""
        return datetime.now(UTC).isoformat()

    @staticmethod
    def parse_iso_timestamp(iso_string: str) -> datetime:
        """Parse ISO 8601 timestamp string."""
        try:
            return datetime.fromisoformat(iso_string)
        except ValueError as e:
            msg = f"Invalid ISO timestamp: {iso_string}"
            raise ValueError(msg) from e
