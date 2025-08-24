"""Video utilities for metadata extraction, validation, and analysis."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mivideoeditor.core.models import ValidationResult
from mivideoeditor.utils.time import TimeUtils

try:
    from mivideoeditor.utils.system import SystemUtils

    SYSTEM_UTILS_AVAILABLE = True
except ImportError:
    SYSTEM_UTILS_AVAILABLE = False
    SystemUtils = None

logger = logging.getLogger(__name__)

HD_HEIGHT = 1080
UHD_HEIGHT = 2160


@dataclass
class VideoInfo:
    """Container for video file metadata and properties."""

    file_path: Path
    duration: float  # seconds
    width: int
    height: int
    frame_rate: float
    total_frames: int
    bit_rate: int | None = None
    codec: str | None = None
    format_name: str | None = None
    file_size: int | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    creation_time: str | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height) tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def megapixels(self) -> float:
        """Calculate total megapixels."""
        return (self.width * self.height) / 1_000_000

    @property
    def duration_formatted(self) -> str:
        """Get duration in HH:MM:SS.mmm format."""
        return TimeUtils.format_duration(self.duration, "hms", 3)

    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024) if self.file_size else 0.0

    def is_high_resolution(self) -> bool:
        """Check if video is considered high resolution (>= 1080p)."""
        return self.height >= HD_HEIGHT

    def is_4k(self) -> bool:
        """Check if video is 4K resolution."""
        return self.height >= UHD_HEIGHT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "total_frames": self.total_frames,
            "bit_rate": self.bit_rate,
            "codec": self.codec,
            "format_name": self.format_name,
            "file_size": self.file_size,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "creation_time": self.creation_time,
            "aspect_ratio": self.aspect_ratio,
            "megapixels": self.megapixels,
            "duration_formatted": self.duration_formatted,
            "file_size_mb": self.file_size_mb,
            "is_high_resolution": self.is_high_resolution(),
            "is_4k": self.is_4k(),
        }


class VideoUtils:
    """Utilities for video file analysis and processing."""

    @staticmethod
    def _raise_runtime_error(message: str, error: Exception | None = None) -> None:
        """Raise RuntimeError with logging."""
        logger.error(message)
        if error:
            raise RuntimeError(message) from error
        raise RuntimeError(message)

    @staticmethod
    def _raise_import_error(message: str) -> None:
        """Raise ImportError with logging."""
        logger.error(message)
        raise ImportError(message)

    @staticmethod
    def get_video_info(video_path: Path) -> VideoInfo:
        """Extract comprehensive video metadata using FFprobe."""
        if not video_path.exists():
            msg = f"Video file not found: {video_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Check if ffprobe is available
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            msg = "FFprobe not found. Please install FFmpeg."
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            # Run ffprobe to get video information
            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                error_msg = (
                    result.stderr.strip() if result.stderr else "Unknown FFprobe error"
                )
                msg = f"FFprobe failed: {error_msg}"
                VideoUtils._raise_runtime_error(msg)

            probe_data = json.loads(result.stdout)

        except subprocess.TimeoutExpired as e:
            msg = "FFprobe timed out while analyzing video"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Failed to parse FFprobe output: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"FFprobe execution failed: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        # Extract video stream information
        video_stream = None
        audio_stream = None

        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if not video_stream:
            msg = f"No video stream found in file: {video_path}"
            logger.error(msg)
            raise ValueError(msg)

        # Extract format information
        format_info = probe_data.get("format", {})

        # Parse video properties
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        if width <= 0 or height <= 0:
            msg = f"Invalid video dimensions: {width}x{height}"
            logger.error(msg)
            raise ValueError(msg)

        # Parse duration (try multiple sources)
        duration = 0.0
        duration_sources = [
            video_stream.get("duration"),
            format_info.get("duration"),
        ]

        for duration_str in duration_sources:
            if duration_str:
                try:
                    duration = float(duration_str)
                    if duration > 0:
                        break
                except (ValueError, TypeError):
                    continue

        if duration <= 0:
            msg = "Could not determine valid video duration"
            logger.error(msg)
            raise ValueError(msg)

        # Parse frame rate
        frame_rate_str = video_stream.get("r_frame_rate", "0/1")
        try:
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                frame_rate = float(num) / float(den) if float(den) != 0 else 0.0
            else:
                frame_rate = float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            frame_rate = 30.0  # Default fallback

        # Calculate total frames
        total_frames = int(duration * frame_rate) if frame_rate > 0 else 0

        # Extract additional metadata
        bit_rate = None
        if format_info.get("bit_rate"):
            with suppress(ValueError, TypeError):
                bit_rate = int(format_info["bit_rate"])

        codec = video_stream.get("codec_name")
        format_name = format_info.get("format_name")

        # File size
        file_size = None
        with suppress(OSError):
            file_size = video_path.stat().st_size

        # Audio information
        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None

        # Creation time
        creation_time = format_info.get("tags", {}).get("creation_time")

        return VideoInfo(
            file_path=video_path,
            duration=duration,
            width=width,
            height=height,
            frame_rate=frame_rate,
            total_frames=total_frames,
            bit_rate=bit_rate,
            codec=codec,
            format_name=format_name,
            file_size=file_size,
            has_audio=has_audio,
            audio_codec=audio_codec,
            creation_time=creation_time,
        )

    @staticmethod
    def validate_video_file(video_path: Path) -> ValidationResult:
        """Comprehensive validation of video file."""
        result = ValidationResult(is_valid=True)

        # Basic file validation
        if not video_path.exists():
            result.add_error(f"Video file not found: {video_path}")
            return result

        if not video_path.is_file():
            result.add_error(f"Path is not a file: {video_path}")
            return result

        # File extension check
        supported_extensions = {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
            ".flv",
            ".wmv",
        }
        if video_path.suffix.lower() not in supported_extensions:
            result.add_warning(f"Unusual video extension: {video_path.suffix}")

        # File size checks
        try:
            file_size = video_path.stat().st_size

            if file_size == 0:
                result.add_error("Video file is empty")
                return result
            if file_size < 1024:  # Less than 1KB
                result.add_warning(f"Very small video file: {file_size} bytes")
            elif file_size > 50 * 1024 * 1024 * 1024:  # Greater than 50GB
                result.add_warning(
                    f"Very large video file: {file_size / (1024**3):.1f}GB"
                )

        except OSError as e:
            result.add_error(f"Cannot access file: {e}")
            return result

        # Try to extract video metadata
        try:
            video_info = VideoUtils.get_video_info(video_path)
            result.context["video_info"] = video_info.to_dict()

            # Validate video properties
            validation_result = VideoUtils._validate_video_properties(video_info)
            result = result.merge(validation_result)

        except FileNotFoundError:
            result.add_error("Video file not found during analysis")
        except RuntimeError as e:
            if "FFprobe not found" in str(e):
                result.add_error("FFprobe not available for video analysis")
            else:
                result.add_error(f"Video analysis failed: {e}")
        except ValueError as e:
            result.add_error(f"Invalid video file: {e}")
        except (OSError, TypeError) as e:
            result.add_error(f"Unexpected error analyzing video: {e}")

        return result

    @staticmethod
    def _validate_video_properties(video_info: VideoInfo) -> ValidationResult:
        """Validate video properties for processing compatibility."""
        result = ValidationResult(is_valid=True)

        # Duration validation
        if video_info.duration < 0.1:
            result.add_warning(f"Very short video: {video_info.duration:.3f}s")
        elif video_info.duration > 4 * 3600:  # 4 hours
            result.add_warning(f"Very long video: {video_info.duration_formatted}")

        # Resolution validation
        if video_info.width < 320 or video_info.height < 240:
            result.add_warning(
                f"Very low resolution: {video_info.width}x{video_info.height}"
            )
        elif video_info.width > 7680 or video_info.height > 4320:  # 8K
            result.add_warning(
                f"Very high resolution: {video_info.width}x{video_info.height}"
            )

        # Aspect ratio validation
        aspect_ratio = video_info.aspect_ratio
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            result.add_warning(f"Unusual aspect ratio: {aspect_ratio:.2f}")

        # Frame rate validation
        if video_info.frame_rate < 10:
            result.add_warning(f"Low frame rate: {video_info.frame_rate:.1f}fps")
        elif video_info.frame_rate > 120:
            result.add_warning(f"High frame rate: {video_info.frame_rate:.1f}fps")

        # Bit rate validation (if available)
        if video_info.bit_rate:
            mbps = video_info.bit_rate / 1_000_000
            if mbps < 1:
                result.add_warning(f"Low bitrate: {mbps:.1f}Mbps")
            elif mbps > 100:
                result.add_warning(f"High bitrate: {mbps:.1f}Mbps")

        # Codec validation
        supported_codecs = {"h264", "h265", "hevc", "vp8", "vp9", "av1"}
        if video_info.codec and video_info.codec.lower() not in supported_codecs:
            result.add_warning(f"Potentially unsupported codec: {video_info.codec}")

        # Processing complexity estimation
        complexity_score = VideoUtils._calculate_processing_complexity(video_info)
        result.context["processing_complexity"] = complexity_score

        if complexity_score > 8:
            result.add_warning(
                "High processing complexity - consider using chunked processing"
            )
        elif complexity_score > 6:
            result.add_warning("Moderate processing complexity")

        return result

    @staticmethod
    def _calculate_processing_complexity(video_info: VideoInfo) -> float:
        """Calculate processing complexity score (0-10 scale)."""
        score = 0.0

        # Resolution contribution (0-3 points)
        megapixels = video_info.megapixels
        if megapixels > 8:  # 4K+
            score += 3
        elif megapixels > 2:  # 1080p+
            score += 2
        elif megapixels > 1:  # 720p+
            score += 1

        # Frame rate contribution (0-2 points)
        if video_info.frame_rate > 60:
            score += 2
        elif video_info.frame_rate > 30:
            score += 1

        # Duration contribution (0-3 points)
        if video_info.duration > 2 * 3600:  # > 2 hours
            score += 3
        elif video_info.duration > 3600:  # > 1 hour
            score += 2
        elif video_info.duration > 1800:  # > 30 minutes
            score += 1

        # Bit rate contribution (0-2 points)
        if video_info.bit_rate:
            mbps = video_info.bit_rate / 1_000_000
            if mbps > 50:
                score += 2
            elif mbps > 20:
                score += 1

        return min(score, 10.0)

    @staticmethod
    def estimate_processing_time(
        video_info: VideoInfo, processing_mode: str = "balanced"
    ) -> dict[str, Any]:
        """Estimate processing time for video based on properties."""
        try:
            if not SYSTEM_UTILS_AVAILABLE:
                msg = "SystemUtils not available"
                VideoUtils._raise_import_error(msg)
            system_info = SystemUtils.get_system_info()
            gpu_info = SystemUtils.get_gpu_info()
        except ImportError:
            system_info = {}
            gpu_info = {}

        # Base processing speed (realtime multiplier)
        mode_multipliers = {"fast": 5.0, "balanced": 1.5, "high": 0.8, "maximum": 0.4}

        base_multiplier = mode_multipliers.get(processing_mode, 1.5)

        # Complexity adjustments
        complexity_score = VideoUtils._calculate_processing_complexity(video_info)
        complexity_factor = 1.0 - (
            complexity_score / 20.0
        )  # Reduce speed for complexity

        # System performance adjustments
        performance_factor = 1.0

        # CPU cores adjustment
        cpu_cores = system_info.get("resources", {}).get("cpu_cores")
        if isinstance(cpu_cores, int):
            if cpu_cores >= 8:
                performance_factor *= 1.4
            elif cpu_cores >= 4:
                performance_factor *= 1.0
            else:
                performance_factor *= 0.7

        # GPU acceleration adjustment
        if gpu_info.get("available"):
            if gpu_info.get("nvidia"):
                performance_factor *= 2.5  # CUDA acceleration
            elif gpu_info.get("videotoolbox"):
                performance_factor *= 2.0  # VideoToolbox acceleration

        # Calculate final estimate
        effective_multiplier = base_multiplier * complexity_factor * performance_factor
        estimated_seconds = video_info.duration / max(effective_multiplier, 0.1)

        return {
            "estimated_processing_time": estimated_seconds,
            "estimated_time_formatted": TimeUtils.format_duration(
                estimated_seconds, "hms", 0
            ),
            "effective_multiplier": effective_multiplier,
            "complexity_score": complexity_score,
            "performance_factors": {
                "mode_multiplier": base_multiplier,
                "complexity_factor": complexity_factor,
                "performance_factor": performance_factor,
            },
            "recommendations": VideoUtils._get_processing_recommendations(
                video_info, complexity_score, effective_multiplier
            ),
        }

    @staticmethod
    def _get_processing_recommendations(
        video_info: VideoInfo, complexity_score: float, effective_multiplier: float
    ) -> list[str]:
        """Generate processing recommendations based on video properties."""
        recommendations = []

        # Duration-based recommendations
        if video_info.duration > 2 * 3600:  # > 2 hours
            recommendations.append(
                "Consider chunked processing for videos longer than 2 hours"
            )

        # Resolution-based recommendations
        if video_info.is_4k():
            recommendations.append(
                "4K video detected - ensure sufficient system memory"
            )
            recommendations.append(
                "Consider reducing quality mode for faster processing"
            )

        # Complexity recommendations
        if complexity_score > 7:
            recommendations.append(
                "High complexity video - allow extra processing time"
            )
            if effective_multiplier < 1.0:
                recommendations.append("Consider 'fast' mode for initial testing")

        # Performance recommendations
        if effective_multiplier < 0.5:
            recommendations.append(
                "Slow processing expected - consider upgrading hardware"
            )
        elif effective_multiplier > 5.0:
            recommendations.append(
                "Fast processing expected - can use higher quality modes"
            )

        # File size recommendations
        if video_info.file_size_mb > 10000:  # > 10GB
            recommendations.append("Large file detected - ensure sufficient disk space")

        # Hardware acceleration recommendations
        if SYSTEM_UTILS_AVAILABLE:
            gpu_info = SystemUtils.get_gpu_info()
            if not gpu_info.get("available"):
                recommendations.append(
                    "No hardware acceleration detected - processing may be slow"
                )
            elif gpu_info.get("videotoolbox"):
                recommendations.append(
                    "VideoToolbox acceleration available - enable for better performance"
                )
        else:
            recommendations.append("System info unavailable")

        if not recommendations:
            recommendations.append("Video appears suitable for standard processing")

        return recommendations

    @staticmethod
    def compare_video_files(video_path1: Path, video_path2: Path) -> dict[str, Any]:
        """Compare two video files and return compatibility analysis."""
        try:
            info1 = VideoUtils.get_video_info(video_path1)
            info2 = VideoUtils.get_video_info(video_path2)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as e:
            return {"error": f"Failed to analyze videos: {e}"}

        comparison = {
            "video1": info1.to_dict(),
            "video2": info2.to_dict(),
            "differences": {},
            "compatibility": {
                "resolution_match": info1.resolution == info2.resolution,
                "frame_rate_match": abs(info1.frame_rate - info2.frame_rate) < 0.1,
                "codec_match": info1.codec == info2.codec,
                "format_match": info1.format_name == info2.format_name,
            },
        }

        # Calculate differences
        comparison["differences"] = {
            "duration_diff": abs(info1.duration - info2.duration),
            "resolution_diff": (info1.width - info2.width, info1.height - info2.height),
            "frame_rate_diff": info1.frame_rate - info2.frame_rate,
            "size_diff": (info1.file_size or 0) - (info2.file_size or 0),
        }

        # Overall compatibility score (0-100)
        compatibility_score = 100

        if not comparison["compatibility"]["resolution_match"]:
            compatibility_score -= 30
        if not comparison["compatibility"]["frame_rate_match"]:
            compatibility_score -= 20
        if not comparison["compatibility"]["codec_match"]:
            compatibility_score -= 20
        if not comparison["compatibility"]["format_match"]:
            compatibility_score -= 10

        comparison["compatibility_score"] = max(0, compatibility_score)

        return comparison

    @staticmethod
    def suggest_processing_settings(video_info: VideoInfo) -> dict[str, Any]:
        """Suggest optimal processing settings based on video properties."""
        settings = {
            "quality_mode": "balanced",
            "frame_step": 10,
            "chunk_size": None,
            "hardware_acceleration": "auto",
            "reasoning": [],
        }

        # Quality mode based on resolution and complexity
        complexity = VideoUtils._calculate_processing_complexity(video_info)

        if complexity > 7 or video_info.is_4k():
            settings["quality_mode"] = "fast"
            settings["reasoning"].append("High complexity/4K video - using fast mode")
        elif complexity < 3 and video_info.duration < 300:  # < 5 minutes
            settings["quality_mode"] = "high"
            settings["reasoning"].append(
                "Low complexity short video - using high quality"
            )

        # Frame step based on frame rate and duration
        if video_info.frame_rate > 60:
            settings["frame_step"] = 15
            settings["reasoning"].append("High frame rate - increased frame step")
        elif video_info.duration < 300:  # < 5 minutes
            settings["frame_step"] = 5
            settings["reasoning"].append(
                "Short video - decreased frame step for accuracy"
            )

        # Chunked processing for long videos
        if video_info.duration > 3600:  # > 1 hour
            settings["chunk_size"] = 300  # 5 minute chunks
            settings["reasoning"].append("Long video - enabling chunked processing")

            # Hardware acceleration based on system
        if SYSTEM_UTILS_AVAILABLE:
            gpu_info = SystemUtils.get_gpu_info()
            if gpu_info.get("videotoolbox"):
                settings["hardware_acceleration"] = "videotoolbox"
                settings["reasoning"].append("VideoToolbox available")
            elif gpu_info.get("nvidia"):
                settings["hardware_acceleration"] = "cuda"
                settings["reasoning"].append("NVIDIA GPU available")
            else:
                settings["hardware_acceleration"] = "none"
                settings["reasoning"].append("No hardware acceleration available")
        else:
            settings["reasoning"].append("System info unavailable")

        return settings
