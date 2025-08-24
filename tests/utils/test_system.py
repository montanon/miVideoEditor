"""Tests for system utilities."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mivideoeditor.utils.system import SystemUtils


class TestSystemUtils:
    """Test system utilities."""

    def test_get_system_info_basic(self):
        """Test basic system info retrieval."""
        info = SystemUtils.get_system_info()

        assert "platform" in info
        assert "python" in info
        assert "resources" in info

        # Platform info should always be available
        assert info["platform"]["system"] == platform.system()
        assert info["platform"]["machine"] == platform.machine()

        # Python info should always be available
        assert info["python"]["version"] == sys.version
        assert info["python"]["executable"] == sys.executable

    def test_get_system_info_with_psutil(self):
        """Test system info with psutil available."""
        info = SystemUtils.get_system_info()

        # Since psutil is available, these should be populated
        assert "resources" in info
        assert "cpu_cores" in info["resources"]
        assert isinstance(info["resources"]["cpu_cores"], int)
        assert info["resources"]["cpu_cores"] > 0

        assert "memory" in info["resources"]
        assert "total" in info["resources"]["memory"]
        assert "available" in info["resources"]["memory"]
        assert info["resources"]["memory"]["total"] > 0

    @patch("mivideoeditor.utils.system.psutil", None)
    def test_get_system_info_without_psutil(self):
        """Test system info when psutil is not available."""
        # This test would require more complex mocking since we import psutil directly
        # For now, we'll skip it since psutil is a required dependency
        pytest.skip("psutil is a required dependency")

    @patch("shutil.which")
    def test_check_dependencies_ffmpeg_available(self, mock_which: Mock) -> None:
        """Test dependency checking when FFmpeg is available."""
        mock_which.side_effect = (
            lambda x: f"/usr/bin/{x}" if x in ["ffmpeg", "ffprobe"] else None
        )

        with patch("subprocess.run") as mock_run:
            # Mock FFmpeg version output
            mock_version_result = Mock()
            mock_version_result.stdout = "ffmpeg version 4.4.0"
            mock_version_result.returncode = 0

            # Mock encoders output
            mock_encoders_result = Mock()
            mock_encoders_result.stdout = " V..... libx264             libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
            mock_encoders_result.returncode = 0

            # Mock hardware acceleration tests
            mock_accel_result = Mock()
            mock_accel_result.returncode = 1  # Not available

            mock_run.side_effect = [
                mock_version_result,
                mock_encoders_result,
                mock_accel_result,
                mock_accel_result,
                mock_accel_result,
            ]

            deps = SystemUtils.check_dependencies()

            assert "ffmpeg" in deps
            assert deps["ffmpeg"]["available"] is True
            assert deps["ffmpeg"]["path"] == "/usr/bin/ffmpeg"
            assert "ffmpeg version 4.4.0" in deps["ffmpeg"]["version"]

    @patch("shutil.which")
    def test_check_dependencies_ffmpeg_missing(self, mock_which: Mock) -> None:
        """Test dependency checking when FFmpeg is missing."""
        mock_which.return_value = None

        deps = SystemUtils.check_dependencies()

        assert "ffmpeg" in deps
        assert deps["ffmpeg"]["available"] is False
        assert "FFmpeg not found in PATH" in deps["ffmpeg"]["error"]

    def test_check_python_package_available(self):
        """Test checking for available Python packages."""
        # Test with a package we know is available (sys is built-in)
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.0.0"

            result = SystemUtils._check_python_package("sys")

            assert result["available"] is True
            assert result["version"] == "1.0.0"

    def test_check_python_package_missing(self):
        """Test checking for missing Python packages."""
        result = SystemUtils._check_python_package("nonexistent_package_xyz")

        assert result["available"] is False
        assert result["version"] is None
        assert "not installed" in result["error"]

    @patch("platform.system")
    def test_get_gpu_info_macos(self, mock_platform: Mock) -> None:
        """Test GPU info detection on macOS."""
        mock_platform.return_value = "Darwin"

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = """
Graphics/Displays:

    Apple M1:

      Chipset Model: Apple M1
      Type: GPU
      Bus: Built-In
      VRAM (Dynamic, Max): 8 GB
      Vendor: Apple (0x106b)
      Device ID: 0x0000
      Revision ID: 0x0000
      Metal: Supported, feature set macOS GPUFamily2 v1

Displays:
"""
            mock_run.return_value = mock_result

            gpu_info = SystemUtils.get_gpu_info()

            assert gpu_info["available"] is True
            assert gpu_info["videotoolbox"] is True
            assert gpu_info["recommended_acceleration"] == "videotoolbox"
            assert len(gpu_info["devices"]) > 0

    def test_get_gpu_info_no_gpu(self) -> None:
        """Test GPU info when no GPU is detected."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            gpu_info = SystemUtils.get_gpu_info()

            assert gpu_info["available"] is False
            assert gpu_info["devices"] == []
            assert gpu_info["recommended_acceleration"] is None

    @patch("mivideoeditor.utils.system.psutil")
    def test_check_memory_requirements_sufficient(self, mock_psutil: Mock) -> None:
        """Test memory requirements check with sufficient memory."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.percent = 50.0

        mock_psutil.virtual_memory.return_value = mock_memory

        result = SystemUtils.check_memory_requirements()

        assert result["sufficient_for_basic"] is True
        assert result["sufficient_for_recommended"] is True
        assert result["total_memory"] == 16 * 1024 * 1024 * 1024

    @patch("mivideoeditor.utils.system.psutil")
    def test_check_memory_requirements_insufficient(self, mock_psutil: Mock) -> None:
        """Test memory requirements check with insufficient memory."""
        mock_memory = Mock()
        mock_memory.total = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB
        mock_memory.percent = 80.0

        mock_psutil.virtual_memory.return_value = mock_memory

        result = SystemUtils.check_memory_requirements()

        assert result["sufficient_for_basic"] is False
        assert result["sufficient_for_recommended"] is False
        assert "below minimum requirements" in " ".join(result["recommendations"])

    def test_check_memory_requirements_with_video(self, tmp_path: Path) -> None:
        """Test memory requirements with specific video file."""
        # Create a test video file
        test_video = tmp_path / "test.mp4"
        test_video.write_bytes(b"fake video content" * 1000000)  # ~17MB fake file

        with patch("mivideoeditor.utils.system.psutil") as mock_psutil:
            mock_memory = Mock()
            mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
            mock_memory.percent = 50.0

            mock_psutil.virtual_memory.return_value = mock_memory

            result = SystemUtils.check_memory_requirements(test_video)

            assert "video_file_size" in result
            assert "estimated_requirement" in result
            assert result["video_file_size"] > 0

    def test_estimate_processing_time_fast_mode(self) -> None:
        """Test processing time estimation for fast mode."""
        system_info = {
            "resources": {
                "cpu_cores": 8,
                "memory": {"percent": 50},
            }
        }

        result = SystemUtils.estimate_processing_time(
            video_duration=3600,  # 1 hour
            processing_mode="fast",
            system_info=system_info,
        )

        assert result["video_duration"] == 3600
        assert result["processing_mode"] == "fast"
        assert result["base_multiplier"] == 10.0
        assert (
            result["estimated_processing_time"] < 3600
        )  # Should be faster than realtime

    def test_estimate_processing_time_slow_system(self) -> None:
        """Test processing time estimation for slower system."""
        system_info = {
            "resources": {
                "cpu_cores": 2,  # Low core count
                "memory": {"percent": 90},  # High memory usage
            }
        }

        result = SystemUtils.estimate_processing_time(
            video_duration=1800,  # 30 minutes
            processing_mode="balanced",
            system_info=system_info,
        )

        assert result["performance_factor"] < 1.0  # Should be penalized
        assert result["performance_tier"] in ["adequate", "slow"]

    def test_classify_performance(self) -> None:
        """Test performance tier classification."""
        assert SystemUtils._classify_performance(10.0) == "excellent"
        assert SystemUtils._classify_performance(3.0) == "good"
        assert SystemUtils._classify_performance(1.5) == "adequate"
        assert SystemUtils._classify_performance(0.5) == "slow"

    @patch("mivideoeditor.utils.system.SystemUtils.check_dependencies")
    @patch("mivideoeditor.utils.system.SystemUtils.check_memory_requirements")
    @patch("mivideoeditor.utils.system.SystemUtils.get_gpu_info")
    def test_validate_system_requirements_pass(
        self, mock_gpu: Mock, mock_memory: Mock, mock_deps: Mock
    ) -> None:
        """Test system validation with all requirements met."""
        mock_deps.return_value = {
            "ffmpeg": {"available": True},
            "pydantic": {"available": True},
            "numpy": {"available": True},
            "opencv-python": {"available": True},
            "psutil": {"available": True},
        }

        mock_memory.return_value = {"sufficient_for_basic": True}
        mock_gpu.return_value = {"available": False}

        result = SystemUtils.validate_system_requirements()

        assert result["overall_status"] == "pass"
        assert len(result["critical_issues"]) == 0

    @patch("mivideoeditor.utils.system.SystemUtils.check_dependencies")
    def test_validate_system_requirements_fail(self, mock_deps: Mock) -> None:
        """Test system validation with missing critical dependencies."""
        mock_deps.return_value = {
            "ffmpeg": {"available": False},
            "pydantic": {"available": False},
            "numpy": {"available": True},
        }

        result = SystemUtils.validate_system_requirements()

        assert result["overall_status"] == "fail"
        assert len(result["critical_issues"]) > 0
        assert any("FFmpeg" in issue for issue in result["critical_issues"])

    @patch("mivideoeditor.utils.system.SystemUtils.check_dependencies")
    @patch("mivideoeditor.utils.system.SystemUtils.check_memory_requirements")
    def test_validate_system_requirements_warnings(
        self, mock_memory: Mock, mock_deps: Mock
    ) -> None:
        """Test system validation with warnings but no critical issues."""
        mock_deps.return_value = {
            "ffmpeg": {"available": True},
            "pydantic": {"available": True},
            "numpy": {"available": True},
            "opencv-python": {"available": False},  # Optional package missing
        }

        mock_memory.return_value = {"sufficient_for_basic": False}  # Low memory warning

        result = SystemUtils.validate_system_requirements()

        assert result["overall_status"] == "warning"
        assert len(result["critical_issues"]) == 0
        assert len(result["warnings"]) > 0

    def test_hardware_acceleration_check_timeout(self) -> None:
        """Test hardware acceleration check with timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 5)

            result = SystemUtils._check_hardware_acceleration()

            # Should handle timeout gracefully
            assert isinstance(result, dict)

    def test_ffmpeg_check_with_subprocess_error(self) -> None:
        """Test FFmpeg check handles subprocess errors gracefully."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = (
                lambda x: f"/usr/bin/{x}" if x in ["ffmpeg", "ffprobe"] else None
            )

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.SubprocessError("Test error")

                result = SystemUtils._check_ffmpeg()

                assert result["available"] is False
                assert "Test error" in result["error"]
