"""Storage module - Data persistence and management."""

from .annotation_service import AnnotationService
from .cache_manager import CacheConfig, CacheManager
from .file_manager import FileManager, FileManagerConfig, FileManagerError
from .models import (
    AnnotationRecord,
    DetectionRecord,
    ModelRecord,
    ProcessingJobRecord,
    TimelineRecord,
    VideoRecord,
)
from .service import StorageConfig, StorageError, StorageService
from .timeline_service import TimelineService

__all__ = [
    # Caching
    "CacheConfig",
    "CacheManager",
    # Core service classes
    "StorageConfig",
    "StorageError",
    "StorageService",
    # Data models
    "AnnotationRecord",
    "DetectionRecord",
    "ModelRecord",
    "ProcessingJobRecord",
    "TimelineRecord",
    "VideoRecord",
    # File management
    "FileManager",
    "FileManagerConfig",
    "FileManagerError",
    # Specialized services
    "AnnotationService",
    "TimelineService",
]
