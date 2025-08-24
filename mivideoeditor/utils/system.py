"""System utilities for dependency checking and performance monitoring."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class SystemUtils:
    """Utilities for system information and dependency management."""

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get comprehensive system information."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk_usage = psutil.disk_usage("/")

            return {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "implementation": platform.python_implementation(),
                },
                "resources": {
                    "cpu_cores": cpu_count,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                    },
                    "disk": {
                        "total": disk_usage.total,
                        "used": disk_usage.used,
                        "free": disk_usage.free,
                        "percent": (disk_usage.used / disk_usage.total) * 100,
                    },
                },
            }
        except ImportError:
            msg = "psutil not available, returning basic system info"
            logger.warning(msg)
            return {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "implementation": platform.python_implementation(),
                },
                "resources": {
                    "cpu_cores": "unknown",
                    "memory": {"status": "psutil_required"},
                    "disk": {"status": "psutil_required"},
                },
            }
        except Exception as e:
            msg = f"Error getting system info: {e}"
            logger.exception(msg)
            return {"error": str(e)}

    @staticmethod
    def check_dependencies() -> dict[str, dict[str, Any]]:
        """Check availability and versions of required dependencies."""
        dependencies = {}

        # Check FFmpeg
        dependencies["ffmpeg"] = SystemUtils._check_ffmpeg()

        # Check Python packages
        python_deps = [
            "pydantic",
            "fastapi",
            "opencv-python",
            "numpy",
            "pillow",
            "psutil",
        ]

        for dep in python_deps:
            dependencies[dep] = SystemUtils._check_python_package(dep)

        return dependencies

    @staticmethod
    def _check_ffmpeg() -> dict[str, Any]:
        """Check FFmpeg availability and capabilities."""
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")

        if not ffmpeg_path or not ffprobe_path:
            return {
                "available": False,
                "error": "FFmpeg not found in PATH",
                "path": None,
                "version": None,
                "encoders": [],
                "decoders": [],
            }

        try:
            # Get version information
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            version_line = result.stdout.split("\n")[0] if result.stdout else ""

            # Get available encoders
            encoders_result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            # Parse encoders for video codecs we care about
            important_encoders = [
                "libx264",
                "libx265",
                "h264_videotoolbox",
                "hevc_videotoolbox",
            ]
            available_encoders = []

            if encoders_result.stdout:
                for line in encoders_result.stdout.split("\n"):
                    for encoder in important_encoders:
                        if encoder in line and "V" in line[:10]:  # Video encoder
                            available_encoders.extend([encoder])

            return {
                "available": True,
                "path": ffmpeg_path,
                "version": version_line,
                "encoders": available_encoders,
                "hardware_acceleration": SystemUtils._check_hardware_acceleration(),
            }

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            return {
                "available": False,
                "error": f"FFmpeg check failed: {e}",
                "path": ffmpeg_path,
                "version": None,
                "encoders": [],
            }

    @staticmethod
    def _check_hardware_acceleration() -> dict[str, bool]:
        """Check for hardware acceleration support."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return {}

        acceleration_types = {
            "videotoolbox": [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=1",
                "-c:v",
                "h264_videotoolbox",
                "-f",
                "null",
                "-",
            ],
            "cuda": [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            "qsv": [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=1",
                "-c:v",
                "h264_qsv",
                "-f",
                "null",
                "-",
            ],
        }

        results = {}
        for accel_type, cmd in acceleration_types.items():
            try:
                result = subprocess.run(
                    [ffmpeg_path] + cmd,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                results[accel_type] = result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                results[accel_type] = False

        return results

    @staticmethod
    def _check_python_package(package_name: str) -> dict[str, Any]:
        """Check if a Python package is available and get its version."""
        try:
            __import__(package_name.replace("-", "_"))
            try:
                version = importlib.metadata.version(package_name)
            except (
                importlib.metadata.PackageNotFoundError,
                ValueError,
                AttributeError,
                ImportError,
            ):
                version = "unknown"

        except ImportError:
            return {
                "available": False,
                "version": None,
                "error": f"Package {package_name} not installed",
            }
        return {
            "available": True,
            "version": version,
        }

    @staticmethod
    def get_gpu_info() -> dict[str, Any]:
        """Get GPU information for hardware acceleration assessment."""
        gpu_info = {
            "available": False,
            "devices": [],
            "recommended_acceleration": None,
        }

        # Check for NVIDIA GPUs
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path:
            try:
                result = subprocess.run(
                    [
                        nvidia_smi_path,
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )

                if result.returncode == 0 and result.stdout.strip():
                    gpu_info["available"] = True
                    gpu_info["nvidia"] = True
                    gpu_info["recommended_acceleration"] = "cuda"

                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            gpu_info["devices"].append(
                                {
                                    "name": parts[0],
                                    "memory_mb": int(parts[1]),
                                    "driver_version": parts[2],
                                    "vendor": "nvidia",
                                }
                            )
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                pass

        # Check for macOS VideoToolbox (Metal)
        if platform.system() == "Darwin":
            system_profiler_path = shutil.which("system_profiler")
            if system_profiler_path:
                try:
                    # Check if we can run a simple VideoToolbox test
                    result = subprocess.run(
                        [system_profiler_path, "SPDisplaysDataType"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )

                    if result.returncode == 0:
                        gpu_info["available"] = True
                        gpu_info["videotoolbox"] = True
                        if not gpu_info["recommended_acceleration"]:
                            gpu_info["recommended_acceleration"] = "videotoolbox"

                        # Parse GPU info from system profiler
                        lines = result.stdout.split("\n")
                        current_gpu = {}
                        for line in lines:
                            stripped_line = line.strip()
                            if "Chipset Model:" in stripped_line:
                                current_gpu["name"] = stripped_line.split(": ", 1)[1]
                                current_gpu["vendor"] = (
                                    "apple"
                                    if "Apple" in current_gpu["name"]
                                    else "intel"
                                )
                            elif "VRAM" in stripped_line and current_gpu:
                                try:
                                    vram_str = stripped_line.split(": ", 1)[1]
                                    if "MB" in vram_str:
                                        current_gpu["memory_mb"] = int(
                                            vram_str.replace(" MB", "")
                                        )
                                    elif "GB" in vram_str:
                                        current_gpu["memory_mb"] = int(
                                            float(vram_str.replace(" GB", "")) * 1024
                                        )
                                except (ValueError, IndexError):
                                    pass
                            elif current_gpu and (
                                "Displays:" in stripped_line or stripped_line == ""
                            ):
                                if "name" in current_gpu:
                                    gpu_info["devices"].append(current_gpu)
                                current_gpu = {}

                        # Add the last GPU if we were parsing one
                        if current_gpu and "name" in current_gpu:
                            gpu_info["devices"].append(current_gpu)

                except (
                    subprocess.TimeoutExpired,
                    subprocess.SubprocessError,
                    FileNotFoundError,
                ):
                    pass

        return gpu_info

    @staticmethod
    def check_memory_requirements(video_path: Path | None = None) -> dict[str, Any]:
        """Check if system has sufficient memory for video processing."""
        try:
            memory = psutil.virtual_memory()

            # Base requirements (in bytes)
            base_requirement = 2 * 1024 * 1024 * 1024  # 2GB base
            recommended_minimum = 4 * 1024 * 1024 * 1024  # 4GB recommended

            assessment = {
                "total_memory": memory.total,
                "available_memory": memory.available,
                "memory_percent_used": memory.percent,
                "base_requirement": base_requirement,
                "recommended_minimum": recommended_minimum,
                "sufficient_for_basic": memory.available >= base_requirement,
                "sufficient_for_recommended": memory.available >= recommended_minimum,
            }

            # Video-specific assessment
            if video_path and video_path.exists():
                try:
                    video_size = video_path.stat().st_size
                    # Rough estimate: need ~3x video file size in RAM for processing
                    estimated_requirement = video_size * 3

                    assessment.update(
                        {
                            "video_file_size": video_size,
                            "estimated_requirement": estimated_requirement,
                            "sufficient_for_video": memory.available
                            >= estimated_requirement,
                            "chunked_processing_recommended": estimated_requirement
                            > memory.available * 0.7,
                        }
                    )
                except OSError as e:
                    assessment["video_assessment_error"] = str(e)

            # Recommendations
            recommendations = []
            if not assessment["sufficient_for_basic"]:
                recommendations.append("System memory below minimum requirements")
            elif not assessment["sufficient_for_recommended"]:
                recommendations.append(
                    "Consider closing other applications for better performance"
                )

            if video_path and assessment.get("chunked_processing_recommended"):
                recommendations.append(
                    "Use chunked processing mode for this video size"
                )

            assessment["recommendations"] = recommendations

        except ImportError:
            return {
                "error": "psutil required for memory assessment",
                "recommendations": ["Install psutil for detailed memory analysis"],
            }
        except (OSError, ValueError, TypeError, RuntimeError) as e:
            return {
                "error": f"Memory check failed: {e}",
                "recommendations": ["Manual memory check recommended"],
            }
        return assessment

    @staticmethod
    def estimate_processing_time(
        video_duration: float,
        processing_mode: str = "balanced",
        system_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Estimate video processing time based on system capabilities."""
        if system_info is None:
            system_info = SystemUtils.get_system_info()

        # Base processing speed multipliers (realtime ratio)
        mode_multipliers = {
            "fast": 10.0,  # 10x realtime
            "balanced": 2.0,  # 2x realtime
            "high": 1.0,  # 1x realtime
            "maximum": 0.5,  # 0.5x realtime
        }

        base_multiplier = mode_multipliers.get(processing_mode, 2.0)

        # Adjust based on system capabilities
        performance_factor = 1.0

        # CPU adjustment
        try:
            cpu_cores = system_info.get("resources", {}).get("cpu_cores", 4)
            if isinstance(cpu_cores, int):
                if cpu_cores >= 8:
                    performance_factor *= 1.3
                elif cpu_cores >= 4:
                    performance_factor *= 1.0
                else:
                    performance_factor *= 0.7
        except (KeyError, TypeError):
            pass

        # Memory adjustment
        try:
            memory_percent = (
                system_info.get("resources", {}).get("memory", {}).get("percent", 50)
            )
            if memory_percent > 80:
                performance_factor *= 0.8  # High memory usage slows processing
            elif memory_percent < 50:
                performance_factor *= 1.1  # Plenty of free memory helps
        except (KeyError, TypeError):
            pass

        # Calculate estimates
        effective_multiplier = base_multiplier * performance_factor
        estimated_time = video_duration / effective_multiplier

        return {
            "video_duration": video_duration,
            "processing_mode": processing_mode,
            "base_multiplier": base_multiplier,
            "performance_factor": performance_factor,
            "effective_multiplier": effective_multiplier,
            "estimated_processing_time": estimated_time,
            "estimated_completion": estimated_time,  # For backward compatibility
            "performance_tier": SystemUtils._classify_performance(effective_multiplier),
        }

    @staticmethod
    def _classify_performance(multiplier: float) -> str:
        """Classify system performance based on effective processing multiplier."""
        if multiplier >= 5.0:
            return "excellent"
        if multiplier >= 2.0:
            return "good"
        if multiplier >= 1.0:
            return "adequate"
        return "slow"

    @staticmethod
    def validate_system_requirements() -> dict[str, Any]:
        """Comprehensive system validation for video processing."""
        validation_result = {
            "overall_status": "unknown",
            "checks": {},
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check dependencies
        deps = SystemUtils.check_dependencies()
        ffmpeg_available = deps.get("ffmpeg", {}).get("available", False)

        validation_result["checks"]["ffmpeg"] = {
            "status": "pass" if ffmpeg_available else "fail",
            "message": "FFmpeg available" if ffmpeg_available else "FFmpeg not found",
        }

        if not ffmpeg_available:
            validation_result["critical_issues"].append(
                "FFmpeg is required but not found in PATH"
            )

        # Check Python packages
        required_packages = ["pydantic", "numpy"]
        optional_packages = ["opencv-python", "pillow", "psutil", "fastapi"]

        for package in required_packages:
            available = deps.get(package, {}).get("available", False)
            validation_result["checks"][package] = {
                "status": "pass" if available else "fail",
                "message": f"{package} available"
                if available
                else f"{package} missing",
            }
            if not available:
                validation_result["critical_issues"].append(
                    f"Required package {package} is not installed"
                )

        for package in optional_packages:
            available = deps.get(package, {}).get("available", False)
            validation_result["checks"][package] = {
                "status": "pass" if available else "warning",
                "message": f"{package} available"
                if available
                else f"{package} missing (optional)",
            }
            if not available:
                validation_result["warnings"].append(
                    f"Optional package {package} not installed"
                )

        # Check system resources
        try:
            memory_check = SystemUtils.check_memory_requirements()
            sufficient_memory = memory_check.get("sufficient_for_basic", False)

            validation_result["checks"]["memory"] = {
                "status": "pass" if sufficient_memory else "warning",
                "message": "Sufficient memory"
                if sufficient_memory
                else "Low memory detected",
            }

            if not sufficient_memory:
                validation_result["warnings"].append(
                    "System memory below recommended minimum"
                )
        except (OSError, ValueError, TypeError):
            validation_result["checks"]["memory"] = {
                "status": "unknown",
                "message": "Could not check memory requirements",
            }

        # Determine overall status
        if validation_result["critical_issues"]:
            validation_result["overall_status"] = "fail"
        elif validation_result["warnings"]:
            validation_result["overall_status"] = "warning"
        else:
            validation_result["overall_status"] = "pass"

        # Generate recommendations
        if validation_result["critical_issues"]:
            validation_result["recommendations"].append(
                "Resolve critical issues before proceeding"
            )

        if not ffmpeg_available:
            if platform.system() == "Darwin":
                validation_result["recommendations"].append(
                    "Install FFmpeg via Homebrew: brew install ffmpeg"
                )
            elif platform.system() == "Linux":
                validation_result["recommendations"].append(
                    "Install FFmpeg via package manager: apt install ffmpeg or yum install ffmpeg"
                )
            else:
                validation_result["recommendations"].append(
                    "Download FFmpeg from https://ffmpeg.org/download.html"
                )

        # Hardware acceleration recommendation
        gpu_info = SystemUtils.get_gpu_info()
        if gpu_info.get("available") and gpu_info.get("recommended_acceleration"):
            validation_result["recommendations"].append(
                f"Hardware acceleration available: {gpu_info['recommended_acceleration']}"
            )

        return validation_result
