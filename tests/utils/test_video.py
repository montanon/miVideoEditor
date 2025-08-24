"""Tests for video utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mivideoeditor.utils.video import VideoInfo, VideoUtils


class TestVideoInfo:
    """Test VideoInfo dataclass."""

    def test_video_info_basic_properties(self):
        """Test basic VideoInfo properties."""
        info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
            bit_rate=5000000,
            codec="h264",
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            file_size=500000000,
            has_audio=True,
            audio_codec="aac",
        )

        # Test basic properties
        assert info.resolution == (1920, 1080)
        assert info.aspect_ratio == pytest.approx(16 / 9, rel=1e-3)
        assert info.megapixels == pytest.approx(2.073, rel=1e-3)
        assert info.duration_formatted == "01:00:00.000"
        assert info.file_size_mb == pytest.approx(476.84, rel=1e-2)

    def test_video_info_resolution_checks(self):
        """Test resolution classification methods."""
        # HD video
        hd_info = VideoInfo(
            file_path=Path("hd.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=1800,
        )
        assert hd_info.is_high_resolution() is True
        assert hd_info.is_4k() is False

        # 4K video
        uhd_info = VideoInfo(
            file_path=Path("4k.mp4"),
            duration=60.0,
            width=3840,
            height=2160,
            frame_rate=30.0,
            total_frames=1800,
        )
        assert uhd_info.is_high_resolution() is True
        assert uhd_info.is_4k() is True

        # SD video
        sd_info = VideoInfo(
            file_path=Path("sd.mp4"),
            duration=60.0,
            width=720,
            height=480,
            frame_rate=30.0,
            total_frames=1800,
        )
        assert sd_info.is_high_resolution() is False
        assert sd_info.is_4k() is False

    def test_video_info_edge_cases(self):
        """Test VideoInfo edge cases."""
        # Zero height (should not crash)
        info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=60.0,
            width=1920,
            height=0,
            frame_rate=30.0,
            total_frames=1800,
        )
        assert info.aspect_ratio == 0.0

        # No file size
        info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=1800,
            file_size=None,
        )
        assert info.file_size_mb == 0.0

    def test_video_info_to_dict(self):
        """Test VideoInfo to_dict conversion."""
        info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
            codec="h264",
        )

        data = info.to_dict()

        # Check all required fields are present
        required_fields = [
            "file_path",
            "duration",
            "width",
            "height",
            "frame_rate",
            "total_frames",
            "aspect_ratio",
            "megapixels",
            "duration_formatted",
            "file_size_mb",
            "is_high_resolution",
            "is_4k",
        ]

        for field in required_fields:
            assert field in data

        assert data["file_path"] == "test.mp4"
        assert data["width"] == 1920
        assert data["height"] == 1080


class TestVideoUtils:
    """Test VideoUtils class."""

    @patch("shutil.which")
    def test_get_video_info_success(self, mock_which: Mock) -> None:
        """Test successful video info extraction."""
        mock_which.return_value = "/usr/bin/ffprobe"

        # Mock ffprobe output
        mock_ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "duration": "3600.000000",
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {
                "duration": "3600.000000",
                "bit_rate": "5000000",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"creation_time": "2023-01-01T00:00:00.000000Z"},
            },
        }

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_ffprobe_output)
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.stat") as mock_stat,
            ):
                mock_stat_result = Mock()
                mock_stat_result.st_size = 500000000
                mock_stat.return_value = mock_stat_result

                info = VideoUtils.get_video_info(Path("test.mp4"))

                assert info.width == 1920
                assert info.height == 1080
                assert info.duration == 3600.0
                assert info.frame_rate == 30.0
                assert info.codec == "h264"
                assert info.has_audio is True
                assert info.audio_codec == "aac"
                assert info.creation_time == "2023-01-01T00:00:00.000000Z"

    def test_get_video_info_file_not_found(self) -> None:
        """Test video info extraction with missing file."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            VideoUtils.get_video_info(Path("nonexistent.mp4"))

    @patch("shutil.which")
    def test_get_video_info_ffprobe_missing(self, mock_which: Mock) -> None:
        """Test video info extraction when FFprobe is missing."""
        mock_which.return_value = None

        with (
            patch("pathlib.Path.exists", return_value=True),
            pytest.raises(RuntimeError, match="FFprobe not found"),
        ):
            VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_ffprobe_fails(self, mock_which: Mock) -> None:
        """Test video info extraction when FFprobe fails."""
        mock_which.return_value = "/usr/bin/ffprobe"

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Error: invalid file"
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(RuntimeError, match="FFprobe failed"),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_timeout(self, mock_which: Mock) -> None:
        """Test video info extraction timeout."""
        mock_which.return_value = "/usr/bin/ffprobe"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 30)

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(RuntimeError, match="timed out"),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_invalid_json(self, mock_which: Mock) -> None:
        """Test video info extraction with invalid JSON output."""
        mock_which.return_value = "/usr/bin/ffprobe"

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "invalid json"
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(RuntimeError, match="Failed to parse FFprobe output"),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_no_video_stream(self, mock_which: Mock) -> None:
        """Test video info extraction with no video stream."""
        mock_which.return_value = "/usr/bin/ffprobe"

        mock_output = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            "format": {},
        }

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_output)
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(ValueError, match="No video stream found"),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_invalid_dimensions(self, mock_which: Mock) -> None:
        """Test video info extraction with invalid dimensions."""
        mock_which.return_value = "/usr/bin/ffprobe"

        mock_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 0,  # Invalid
                    "height": 1080,
                    "duration": "60.0",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "60.0"},
        }

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_output)
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(ValueError, match="Invalid video dimensions"),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    @patch("shutil.which")
    def test_get_video_info_no_duration(self, mock_which: Mock) -> None:
        """Test video info extraction with no duration."""
        mock_which.return_value = "/usr/bin/ffprobe"

        mock_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    # No duration
                }
            ],
            "format": {},  # No duration here either
        }

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_output)
            mock_run.return_value = mock_result

            with (
                patch("pathlib.Path.exists", return_value=True),
                pytest.raises(
                    ValueError, match="Could not determine valid video duration"
                ),
            ):
                VideoUtils.get_video_info(Path("test.mp4"))

    def test_validate_video_file_missing(self) -> None:
        """Test video file validation with missing file."""
        result = VideoUtils.validate_video_file(Path("nonexistent.mp4"))

        assert result.is_valid is False
        assert any("not found" in error for error in result.errors)

    def test_validate_video_file_directory(self, tmp_path: Path) -> None:
        """Test video file validation with directory."""
        result = VideoUtils.validate_video_file(tmp_path)

        assert result.is_valid is False
        assert any("is not a file" in error for error in result.errors)

    def test_validate_video_file_extension_warning(self, tmp_path: Path) -> None:
        """Test video file validation with unusual extension."""
        test_file = tmp_path / "video.xyz"
        test_file.write_bytes(b"fake video")

        result = VideoUtils.validate_video_file(test_file)

        assert any("Unusual video extension" in warning for warning in result.warnings)

    def test_validate_video_file_empty(self, tmp_path: Path) -> None:
        """Test video file validation with empty file."""
        test_file = tmp_path / "empty.mp4"
        test_file.write_bytes(b"")

        result = VideoUtils.validate_video_file(test_file)

        assert result.is_valid is False
        assert any("empty" in error for error in result.errors)

    def test_validate_video_file_size_warnings(self, tmp_path: Path) -> None:
        """Test video file validation with size warnings."""
        # Very small file
        small_file = tmp_path / "small.mp4"
        small_file.write_bytes(b"x" * 100)

        result = VideoUtils.validate_video_file(small_file)
        assert any("Very small video file" in warning for warning in result.warnings)

        # Would test very large file but that's impractical in tests

    @patch.object(VideoUtils, "get_video_info")
    def test_validate_video_file_with_info(
        self, mock_get_info: Mock, tmp_path: Path
    ) -> None:
        """Test video file validation with successful info extraction."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video" * 1000)

        # Mock video info
        mock_info = VideoInfo(
            file_path=test_file,
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
            codec="h264",
        )
        mock_get_info.return_value = mock_info

        result = VideoUtils.validate_video_file(test_file)

        assert result.is_valid is True
        assert result.context["duration"] == 3600.0
        assert result.context["width"] == 1920
        assert result.context["height"] == 1080

    def test_validate_video_properties_duration_warnings(self) -> None:
        """Test video property validation duration warnings."""
        # Very short video
        short_info = VideoInfo(
            file_path=Path("short.mp4"),
            duration=0.05,  # 50ms
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=2,
        )

        result = VideoUtils._validate_video_properties(short_info)
        assert any("Very short video" in warning for warning in result.warnings)

        # Very long video
        long_info = VideoInfo(
            file_path=Path("long.mp4"),
            duration=5 * 3600,  # 5 hours
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=540000,
        )

        result = VideoUtils._validate_video_properties(long_info)
        assert any("Very long video" in warning for warning in result.warnings)

    def test_validate_video_properties_resolution_warnings(self) -> None:
        """Test video property validation resolution warnings."""
        # Very low resolution
        low_res_info = VideoInfo(
            file_path=Path("lowres.mp4"),
            duration=60.0,
            width=160,
            height=120,
            frame_rate=30.0,
            total_frames=1800,
        )

        result = VideoUtils._validate_video_properties(low_res_info)
        assert any("Very low resolution" in warning for warning in result.warnings)

        # Very high resolution
        high_res_info = VideoInfo(
            file_path=Path("highres.mp4"),
            duration=60.0,
            width=8192,
            height=4320,
            frame_rate=30.0,
            total_frames=1800,
        )

        result = VideoUtils._validate_video_properties(high_res_info)
        assert any("Very high resolution" in warning for warning in result.warnings)

    def test_validate_video_properties_aspect_ratio_warning(self) -> None:
        """Test video property validation aspect ratio warning."""
        weird_aspect_info = VideoInfo(
            file_path=Path("weird.mp4"),
            duration=60.0,
            width=1920,
            height=600,  # Very wide aspect ratio
            frame_rate=30.0,
            total_frames=1800,
        )

        result = VideoUtils._validate_video_properties(weird_aspect_info)
        assert any("Unusual aspect ratio" in warning for warning in result.warnings)

    def test_validate_video_properties_frame_rate_warnings(self) -> None:
        """Test video property validation frame rate warnings."""
        # Low frame rate
        low_fps_info = VideoInfo(
            file_path=Path("lowfps.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=5.0,
            total_frames=300,
        )

        result = VideoUtils._validate_video_properties(low_fps_info)
        assert any("Low frame rate" in warning for warning in result.warnings)

        # High frame rate
        high_fps_info = VideoInfo(
            file_path=Path("highfps.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=240.0,
            total_frames=14400,
        )

        result = VideoUtils._validate_video_properties(high_fps_info)
        assert any("High frame rate" in warning for warning in result.warnings)

    def test_validate_video_properties_bitrate_warnings(self) -> None:
        """Test video property validation bitrate warnings."""
        # Low bitrate
        low_bitrate_info = VideoInfo(
            file_path=Path("lowbitrate.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=1800,
            bit_rate=500000,  # 0.5 Mbps
        )

        result = VideoUtils._validate_video_properties(low_bitrate_info)
        assert any("Low bitrate" in warning for warning in result.warnings)

        # High bitrate
        high_bitrate_info = VideoInfo(
            file_path=Path("highbitrate.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=1800,
            bit_rate=150000000,  # 150 Mbps
        )

        result = VideoUtils._validate_video_properties(high_bitrate_info)
        assert any("High bitrate" in warning for warning in result.warnings)

    def test_validate_video_properties_codec_warning(self) -> None:
        """Test video property validation unsupported codec warning."""
        unsupported_codec_info = VideoInfo(
            file_path=Path("weird.mp4"),
            duration=60.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=1800,
            codec="weird_codec",
        )

        result = VideoUtils._validate_video_properties(unsupported_codec_info)
        assert any(
            "Potentially unsupported codec" in warning for warning in result.warnings
        )

    def test_calculate_processing_complexity(self) -> None:
        """Test processing complexity calculation."""
        # Low complexity video
        low_complexity_info = VideoInfo(
            file_path=Path("simple.mp4"),
            duration=300.0,  # 5 minutes
            width=1280,
            height=720,
            frame_rate=24.0,
            total_frames=7200,
            bit_rate=2000000,
        )

        complexity = VideoUtils._calculate_processing_complexity(low_complexity_info)
        assert 0 <= complexity <= 10
        assert complexity < 5  # Should be relatively low

        # High complexity video
        high_complexity_info = VideoInfo(
            file_path=Path("complex.mp4"),
            duration=7200.0,  # 2 hours
            width=3840,
            height=2160,  # 4K
            frame_rate=60.0,  # High frame rate
            total_frames=432000,
            bit_rate=80000000,  # High bitrate
        )

        complexity = VideoUtils._calculate_processing_complexity(high_complexity_info)
        assert complexity > 8  # Should be very high

    @patch("mivideoeditor.utils.video.SYSTEM_UTILS_AVAILABLE", True)
    @patch("mivideoeditor.utils.video.SystemUtils")
    def test_estimate_processing_time_with_system_info(
        self, mock_system_utils: Mock
    ) -> None:
        """Test processing time estimation with system info."""
        # Mock system info
        mock_system_utils.get_system_info.return_value = {"resources": {"cpu_cores": 8}}
        mock_system_utils.get_gpu_info.return_value = {
            "available": True,
            "videotoolbox": True,
        }

        video_info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
        )

        result = VideoUtils.estimate_processing_time(video_info, "balanced")

        assert "estimated_processing_time" in result
        assert "estimated_time_formatted" in result
        assert "effective_multiplier" in result
        assert "complexity_score" in result
        assert "performance_factors" in result
        assert "recommendations" in result

        # Should be faster than realtime due to good hardware
        assert result["effective_multiplier"] > 1.0

    @patch("mivideoeditor.utils.video.SYSTEM_UTILS_AVAILABLE", False)
    def test_estimate_processing_time_without_system_info(self) -> None:
        """Test processing time estimation without system info."""
        video_info = VideoInfo(
            file_path=Path("test.mp4"),
            duration=1800.0,  # 30 minutes
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=54000,
        )

        result = VideoUtils.estimate_processing_time(video_info, "fast")

        assert "estimated_processing_time" in result
        assert result["performance_factors"]["performance_factor"] == 1.0  # Default
        assert any(
            "System info unavailable" in rec for rec in result["recommendations"]
        )

    def test_get_processing_recommendations(self) -> None:
        """Test processing recommendations generation."""
        # Long 4K video
        video_info = VideoInfo(
            file_path=Path("big.mp4"),
            duration=10800.0,  # 3 hours
            width=3840,
            height=2160,  # 4K
            frame_rate=30.0,
            total_frames=324000,
            file_size=50 * 1024 * 1024 * 1024,  # 50GB
        )

        recommendations = VideoUtils._get_processing_recommendations(
            video_info, 8.5, 0.3
        )

        # Should have recommendations for long video, 4K, high complexity, slow processing, large file
        rec_text = " ".join(recommendations).lower()
        assert "chunk" in rec_text or "hour" in rec_text
        assert "4k" in rec_text or "memory" in rec_text
        assert "complexity" in rec_text or "time" in rec_text
        assert "slow" in rec_text or "hardware" in rec_text
        assert "large" in rec_text or "disk" in rec_text

    @patch.object(VideoUtils, "get_video_info")
    def test_compare_video_files_success(self, mock_get_info: Mock) -> None:
        """Test successful video file comparison."""
        # Mock video info for both files
        info1 = VideoInfo(
            file_path=Path("video1.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
            codec="h264",
            format_name="mp4",
            file_size=500000000,
        )

        info2 = VideoInfo(
            file_path=Path("video2.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=29.97,  # Slightly different
            total_frames=107892,
            codec="h264",
            format_name="mp4",
            file_size=480000000,
        )

        mock_get_info.side_effect = [info1, info2]

        result = VideoUtils.compare_video_files(Path("video1.mp4"), Path("video2.mp4"))

        assert "video1" in result
        assert "video2" in result
        assert "compatibility" in result
        assert "differences" in result
        assert "compatibility_score" in result

        # Should match on most things
        assert result["compatibility"]["resolution_match"] is True
        assert result["compatibility"]["codec_match"] is True
        assert result["compatibility"]["format_match"] is True
        # Frame rate should be close enough to match
        assert result["compatibility"]["frame_rate_match"] is True

    @patch.object(VideoUtils, "get_video_info")
    def test_compare_video_files_different(self, mock_get_info: Mock) -> None:
        """Test video file comparison with different videos."""
        info1 = VideoInfo(
            file_path=Path("hd.mp4"),
            duration=3600.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            total_frames=108000,
            codec="h264",
            format_name="mp4",
        )

        info2 = VideoInfo(
            file_path=Path("4k.mp4"),
            duration=1800.0,
            width=3840,
            height=2160,
            frame_rate=60.0,
            total_frames=108000,
            codec="h265",
            format_name="mov",
        )

        mock_get_info.side_effect = [info1, info2]

        result = VideoUtils.compare_video_files(Path("hd.mp4"), Path("4k.mp4"))

        # Should not match on most things
        assert result["compatibility"]["resolution_match"] is False
        assert result["compatibility"]["frame_rate_match"] is False
        assert result["compatibility"]["codec_match"] is False
        assert result["compatibility"]["format_match"] is False

        # Low compatibility score
        assert result["compatibility_score"] < 50

    @patch.object(VideoUtils, "get_video_info")
    def test_compare_video_files_error(self, mock_get_info: Mock) -> None:
        """Test video file comparison with error."""
        mock_get_info.side_effect = FileNotFoundError("File not found")

        result = VideoUtils.compare_video_files(
            Path("missing1.mp4"), Path("missing2.mp4")
        )

        assert "error" in result
        assert "File not found" in result["error"]

    @patch("mivideoeditor.utils.video.SYSTEM_UTILS_AVAILABLE", True)
    @patch("mivideoeditor.utils.video.SystemUtils")
    def test_suggest_processing_settings_4k_complex(
        self, mock_system_utils: Mock
    ) -> None:
        """Test processing settings suggestions for 4K complex video."""
        mock_system_utils.get_gpu_info.return_value = {"videotoolbox": True}

        # 4K complex video
        video_info = VideoInfo(
            file_path=Path("4k.mp4"),
            duration=7200.0,  # 2 hours
            width=3840,
            height=2160,
            frame_rate=60.0,
            total_frames=432000,
        )

        settings = VideoUtils.suggest_processing_settings(video_info)

        assert (
            settings["quality_mode"] == "fast"
        )  # Should suggest fast mode for complex video
        assert settings["chunk_size"] == 300  # Should enable chunking for long video
        assert settings["hardware_acceleration"] == "videotoolbox"
        assert any("complexity" in reason.lower() for reason in settings["reasoning"])

    def test_suggest_processing_settings_simple_short(self) -> None:
        """Test processing settings suggestions for simple short video."""
        # Simple short video
        video_info = VideoInfo(
            file_path=Path("simple.mp4"),
            duration=120.0,  # 2 minutes
            width=1280,
            height=720,
            frame_rate=24.0,
            total_frames=2880,
        )

        settings = VideoUtils.suggest_processing_settings(video_info)

        assert (
            settings["quality_mode"] == "high"
        )  # Should suggest high quality for simple video
        assert settings["frame_step"] == 5  # Should decrease frame step for short video
        assert settings["chunk_size"] is None  # No chunking needed for short video

    def test_suggest_processing_settings_high_frame_rate(self) -> None:
        """Test processing settings suggestions for high frame rate video."""
        video_info = VideoInfo(
            file_path=Path("hfr.mp4"),
            duration=1800.0,  # 30 minutes
            width=1920,
            height=1080,
            frame_rate=120.0,  # High frame rate
            total_frames=216000,
        )

        settings = VideoUtils.suggest_processing_settings(video_info)

        assert (
            settings["frame_step"] == 15
        )  # Should increase frame step for high frame rate
        assert any("frame rate" in reason.lower() for reason in settings["reasoning"])

    def test_error_handling_patterns(self) -> None:
        """Test error handling patterns in VideoUtils."""
        # Test _raise_runtime_error method
        with pytest.raises(RuntimeError, match="Test error"):
            VideoUtils._raise_runtime_error("Test error")

        # Test _raise_runtime_error with cause
        original_error = ValueError("Original")
        with pytest.raises(RuntimeError, match="Test error") as exc_info:
            VideoUtils._raise_runtime_error("Test error", original_error)
        assert exc_info.value.__cause__ == original_error
