"""File management service for organized storage of videos, images, and models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from mivideoeditor.core.models import ValidationResult

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


class FileManagerConfig(BaseModel):
    """Configuration for file management operations."""

    max_image_dimension: int = Field(
        default=2048, gt=0, description="Maximum image dimension for optimization"
    )
    max_image_bytes: int = Field(
        default=10 * 1024 * 1024, gt=0, description="Maximum image size in bytes"
    )
    jpeg_quality: int = Field(
        default=85, ge=1, le=100, description="JPEG compression quality"
    )
    png_compression: int = Field(
        default=6, ge=0, le=9, description="PNG compression level"
    )
    temp_file_ttl_hours: int = Field(
        default=24, gt=0, description="TTL for temporary files in hours"
    )
    enable_checksums: bool = Field(
        default=True, description="Generate checksums for file integrity"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }


class FileManagerError(Exception):
    """File manager specific errors."""


class FileManager:
    """Manage file system operations for the video editor."""

    def __init__(self, data_dir: Path, config: FileManagerConfig):
        self.data_dir = Path(data_dir)
        self.config = config

        # Define directory structure
        self.directories = {
            "videos": self.data_dir / "videos",
            "frames": self.data_dir / "annotations" / "frames",
            "models": self.data_dir / "models",
            "timelines": self.data_dir / "timelines",
            "output": self.data_dir / "output",
            "cache": self.data_dir / "cache",
            "temp": self.data_dir / "temp",
            "backups": self.data_dir / "backups",
        }

        self._create_directories()

    def _create_directories(self) -> None:
        """Create the directory structure."""
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep files to preserve empty directories
        for category, dir_path in self.directories.items():
            if category not in ["temp", "cache"]:  # Don't track temp/cache dirs
                gitkeep = dir_path / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.touch()

    def store_video(self, source_path: Path, video_id: str) -> Path:
        """Store video file with organized naming and integrity checking."""
        if not source_path.exists():
            msg = f"Source video file not found: {source_path}"
            raise FileManagerError(msg)

        # Validate file
        self._validate_video_file(source_path)

        # Generate organized destination path
        timestamp = datetime.now(UTC).strftime("%Y%m")
        video_ext = source_path.suffix.lower()
        destination = self.directories["videos"] / timestamp / f"{video_id}{video_ext}"

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Perform atomic copy operation
        return self._atomic_copy(source_path, destination, video_id)

    def store_frame_image(
        self,
        image: np.ndarray,
        annotation_id: str,
        timestamp: float,
        *,
        optimize: bool = True,
    ) -> Path:
        """Store extracted frame image with optimization."""
        if image.size == 0:
            msg = "Cannot store empty image"
            raise FileManagerError(msg)

        # Organize by date for better file management
        date_dir = datetime.fromtimestamp(timestamp, UTC).strftime("%Y%m/%d")
        frame_dir = self.directories["frames"] / date_dir
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp precision
        frame_filename = f"frame_{annotation_id}_{int(timestamp * 1000):010d}.png"
        frame_path = frame_dir / frame_filename

        # Optimize image if requested
        if optimize:
            image = self._optimize_image(image)

        # Save image with error handling
        try:
            success = cv2.imwrite(
                str(frame_path),
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, self.config.png_compression],
            )

            if not success:
                msg = f"Failed to write image file: {frame_path}"
                raise FileManagerError(msg)

            logger.debug(
                "Frame image stored: %s (%s bytes)",
                frame_path,
                frame_path.stat().st_size,
            )

        except Exception as e:
            # Clean up partial file
            if frame_path.exists():
                frame_path.unlink()
            msg = f"Failed to store frame image {annotation_id}: {e}"
            raise FileManagerError(msg) from e
        return frame_path

    def store_model(
        self,
        model_data: bytes,
        model_id: str,
        model_type: str,
        *,
        version: str = "1.0",
    ) -> Path:
        """Store trained model file with versioning."""
        if not model_data:
            msg = "Cannot store empty model data"
            raise FileManagerError(msg)

        # Organize by type and version
        model_dir = self.directories["models"] / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"{model_id}.pkl"
        model_path = model_dir / model_filename

        # Atomic write operation
        temp_path = model_path.with_suffix(".tmp")
        try:
            with temp_path.open("wb") as f:
                f.write(model_data)

            # Generate checksum if enabled
            if self.config.enable_checksums:
                checksum = self._calculate_checksum(temp_path)
                checksum_path = model_path.with_suffix(".sha256")
                with checksum_path.open("w") as f:
                    f.write(f"{checksum}  {model_filename}\n")

            # Atomic rename
            temp_path.rename(model_path)

            logger.info(
                "Model stored: %s -> %s (%s bytes)",
                model_id,
                model_path,
                len(model_data),
            )

        except Exception as e:
            # Clean up partial files
            if temp_path.exists():
                temp_path.unlink()
            msg = f"Failed to store model {model_id}: {e}"
            raise FileManagerError(msg) from e
        return model_path

    def store_timeline_export(
        self,
        timeline_data: dict[str, Any],
        timeline_id: str,
        export_format: str = "json",
    ) -> Path:
        """Store timeline export data."""
        # Organize by date
        date_dir = datetime.now(UTC).strftime("%Y%m")
        timeline_dir = self.directories["timelines"] / date_dir
        timeline_dir.mkdir(parents=True, exist_ok=True)

        timeline_filename = f"{timeline_id}.{export_format}"
        timeline_path = timeline_dir / timeline_filename

        try:
            if export_format == "json":
                with timeline_path.open("w", encoding="utf-8") as f:
                    json.dump(timeline_data, f, indent=2, default=str)
            else:
                msg = f"Unsupported export format: {export_format}"
                raise FileManagerError(msg)

            logger.debug("Timeline exported: %s -> %s", timeline_id, timeline_path)

        except Exception as e:
            if timeline_path.exists():
                timeline_path.unlink()
            msg = f"Failed to export timeline {timeline_id}: {e}"
            raise FileManagerError(msg) from e
        return timeline_path

    def get_file_path(self, file_id: str, category: str) -> Path | None:
        """Get the path to a stored file by searching organized directories."""
        if category not in self.directories:
            msg = f"Unknown file category: {category}"
            raise FileManagerError(msg)

        search_dir = self.directories[category]

        # Search for files matching the ID pattern
        for path in search_dir.rglob(f"*{file_id}*"):
            if path.is_file() and file_id in path.stem:
                return path

        return None

    def delete_file(self, file_path: Path) -> bool:
        """Securely delete a file."""
        if not file_path.exists():
            return True

        try:
            # For sensitive files, perform secure deletion
            if self._is_sensitive_file(file_path):
                return self._secure_delete(file_path)
            file_path.unlink()

        except Exception:
            logger.exception("Failed to delete file %s", file_path)
            return False
        return True

    def cleanup_orphaned_files(self, referenced_files: set[str]) -> dict[str, int]:
        """Remove files not referenced in the provided set."""
        cleanup_stats = {
            "frames_removed": 0,
            "models_removed": 0,
            "timelines_removed": 0,
            "temp_removed": 0,
            "bytes_freed": 0,
        }

        # Clean up frame images
        for frame_file in self.directories["frames"].rglob("*.png"):
            if str(frame_file) not in referenced_files:
                file_size = frame_file.stat().st_size
                if self.delete_file(frame_file):
                    cleanup_stats["frames_removed"] += 1
                    cleanup_stats["bytes_freed"] += file_size

        # Clean up model files
        for model_file in self.directories["models"].rglob("*.pkl"):
            if str(model_file) not in referenced_files:
                file_size = model_file.stat().st_size
                if self.delete_file(model_file):
                    cleanup_stats["models_removed"] += 1
                    cleanup_stats["bytes_freed"] += file_size

                    # Also remove associated checksum file
                    checksum_file = model_file.with_suffix(".sha256")
                    if checksum_file.exists():
                        checksum_file.unlink()

        # Clean up timeline exports
        for timeline_file in self.directories["timelines"].rglob("*.json"):
            if str(timeline_file) not in referenced_files:
                file_size = timeline_file.stat().st_size
                if self.delete_file(timeline_file):
                    cleanup_stats["timelines_removed"] += 1
                    cleanup_stats["bytes_freed"] += file_size

        # Clean up temporary files older than TTL
        cutoff_time = time.time() - (self.config.temp_file_ttl_hours * 3600)
        for temp_file in self.directories["temp"].rglob("*"):
            if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                file_size = temp_file.stat().st_size
                if self.delete_file(temp_file):
                    cleanup_stats["temp_removed"] += 1
                    cleanup_stats["bytes_freed"] += file_size

        logger.info("Cleanup completed: %s", cleanup_stats)
        return cleanup_stats

    def get_storage_usage(self) -> dict[str, Any]:
        """Get detailed storage usage statistics."""
        usage_stats = {}
        total_size = 0
        total_files = 0

        for category, dir_path in self.directories.items():
            if not dir_path.exists():
                continue

            category_size = 0
            category_files = 0

            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    category_size += file_size
                    category_files += 1

            usage_stats[category] = {
                "size_bytes": category_size,
                "size_mb": round(category_size / (1024 * 1024), 2),
                "file_count": category_files,
            }

            total_size += category_size
            total_files += category_files

        usage_stats["total"] = {
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "size_gb": round(total_size / (1024 * 1024 * 1024), 3),
            "file_count": total_files,
        }

        return usage_stats

    def validate_integrity(self) -> ValidationResult:
        """Validate file system integrity and check for corruption."""
        result = ValidationResult(is_valid=True)

        # Check directory structure
        for category, dir_path in self.directories.items():
            if not dir_path.exists():
                result.add_error(f"Missing directory: {category} ({dir_path})")
            elif not dir_path.is_dir():
                result.add_error(f"Path is not a directory: {category} ({dir_path})")

        # Check for checksum mismatches
        checksum_errors = 0
        for checksum_file in self.directories["models"].rglob("*.sha256"):
            try:
                model_file = checksum_file.with_suffix(".pkl")
                if model_file.exists():
                    expected_checksum = checksum_file.read_text().strip().split()[0]
                    actual_checksum = self._calculate_checksum(model_file)

                    if expected_checksum != actual_checksum:
                        checksum_errors += 1
                        result.add_error(f"Checksum mismatch: {model_file}")
                else:
                    result.add_warning(f"Orphaned checksum file: {checksum_file}")
            except (OSError, ValueError) as e:
                result.add_warning(f"Error checking checksum for {checksum_file}: {e}")

        if checksum_errors > 0:
            result.add_error(f"Found {checksum_errors} checksum mismatches")

        # Add usage stats to context
        usage_stats = self.get_storage_usage()
        result.context.update(
            {
                "total_files": usage_stats["total"]["file_count"],
                "total_size_mb": usage_stats["total"]["size_mb"],
            }
        )

        return result

    def _validate_video_file(self, file_path: Path) -> None:
        """Validate video file before storage."""
        if not file_path.is_file():
            msg = f"Not a file: {file_path}"
            raise FileManagerError(msg)

        # Check file extension
        allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        if file_path.suffix.lower() not in allowed_extensions:
            msg = f"Unsupported video format: {file_path.suffix}"
            raise FileManagerError(msg)

        # Check file size
        file_size = file_path.stat().st_size
        max_size = 50 * 1024**3  # 50GB
        if file_size > max_size:
            msg = f"Video file too large: {file_size / (1024**3):.1f}GB"
            raise FileManagerError(msg)

        if file_size == 0:
            msg = "Video file is empty"
            raise FileManagerError(msg)

    def _atomic_copy(self, source: Path, destination: Path, file_id: str) -> Path:
        """Perform atomic file copy with integrity verification."""
        temp_path = destination.with_suffix(".tmp")

        try:
            # Copy file
            shutil.copy2(source, temp_path)

            # Verify integrity if checksums enabled
            if self.config.enable_checksums and not self._verify_file_integrity(
                source, temp_path
            ):
                temp_path.unlink()
                msg = f"File copy verification failed for {file_id}"
                raise FileManagerError(msg)

            # Atomic rename
            temp_path.rename(destination)

            logger.info(
                "File stored: %s -> %s (%s bytes)",
                file_id,
                destination,
                destination.stat().st_size,
            )

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            msg = f"Failed to store file {file_id}: {e}"
            raise FileManagerError(msg) from e
        return destination

    def _verify_file_integrity(self, file1: Path, file2: Path) -> bool:
        """Verify two files are identical using checksums."""
        return self._calculate_checksum(file1) == self._calculate_checksum(file2)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()

        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _optimize_image(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for storage (resize, compress)."""
        height, width = image.shape[:2]

        # Resize if too large (maintain aspect ratio)
        max_dimension = self.config.max_image_dimension
        if max(height, width) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))

            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            logger.debug(
                "Image resized from %sx%s to %sx%s",
                width,
                height,
                new_width,
                new_height,
            )

        # Compress if still too large
        if image.nbytes > self.config.max_image_bytes:
            # Convert to JPEG and back to reduce size
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            success, encoded_img = cv2.imencode(".jpg", image, encode_param)

            if success:
                image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                logger.debug(
                    "Image compressed using JPEG quality %s",
                    self.config.jpeg_quality,
                )

        return image

    def _is_sensitive_file(self, file_path: Path) -> bool:
        """Check if file contains potentially sensitive data."""
        sensitive_dirs = {"frames", "models", "timelines"}
        return any(part in sensitive_dirs for part in file_path.parts)

    def _secure_delete(self, file_path: Path) -> bool:
        """Securely delete file to prevent recovery."""
        try:
            # Get file size
            file_size = file_path.stat().st_size

            # Overwrite with random data (3 passes)
            with file_path.open("r+b") as f:
                for _ in range(3):
                    f.seek(0)
                    f.write(rng.bytes(file_size))
                    f.flush()

                    # Force OS to write to disk
                    os.fsync(f.fileno())

            # Finally delete the file
            file_path.unlink()

        except Exception:
            logger.exception("Secure deletion failed for %s", file_path)
            return False
        return True
