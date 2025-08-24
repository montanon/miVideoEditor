"""Storage service for managing database operations and file storage."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mivideoeditor.core.models import ValidationResult
from mivideoeditor.storage.models import (
    AnnotationRecord,
    VideoRecord,
)

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Configuration for the storage service."""

    database_path: Path = Field(
        default=Path("data/app.db"), description="SQLite database path"
    )
    data_dir: Path = Field(default=Path("data"), description="Data directory root")
    max_connections: int = Field(
        default=20, gt=0, description="Maximum database connections"
    )
    timeout: float = Field(
        default=30.0, gt=0, description="Database timeout in seconds"
    )
    enable_wal_mode: bool = Field(
        default=True, description="Enable WAL mode for better concurrency"
    )
    enable_foreign_keys: bool = Field(
        default=True, description="Enable foreign key constraints"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }


class StorageError(Exception):
    """Base exception for storage operations."""


class DatabaseError(StorageError):
    """Database-specific errors."""


class FileStorageError(StorageError):
    """File storage-specific errors."""


class StorageService:
    """Service for managing database and file storage operations."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, config: StorageConfig):
        self.config = config
        self._connection_pool = {}
        self._pool_lock = threading.Lock()
        self._initialized = False

        # Ensure directories exist
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize the storage service."""
        if self._initialized:
            return

        try:
            # Create database and tables
            with self._get_connection() as conn:
                self._setup_database(conn)
                self._create_tables(conn)
                self._create_indexes(conn)

            self._initialized = True
            logger.info(
                "Storage service initialized with database at %s",
                self.config.database_path,
            )

        except Exception as e:
            msg = "Failed to initialize storage service"
            logger.exception(msg)
            raise DatabaseError(msg) from e

    def close(self) -> None:
        """Close all database connections and cleanup."""
        with self._pool_lock:
            for conn in self._connection_pool.values():
                try:
                    conn.close()
                except Exception:
                    logger.exception("Error closing database connection")
            self._connection_pool.clear()

        self._initialized = False
        logger.info("Storage service closed")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool."""
        thread_id = threading.current_thread().ident

        with self._pool_lock:
            if thread_id not in self._connection_pool:
                conn = sqlite3.connect(
                    str(self.config.database_path),
                    timeout=self.config.timeout,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row
                self._connection_pool[thread_id] = conn
            else:
                conn = self._connection_pool[thread_id]

        try:
            yield conn
        except Exception as e:
            conn.rollback()
            msg = "Database operation failed"
            logger.exception(msg)
            raise DatabaseError(msg) from e

    def _setup_database(self, conn: sqlite3.Connection) -> None:
        """Configure database settings."""
        if self.config.enable_wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")

        if self.config.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all database tables."""
        # Videos table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                duration REAL NOT NULL,
                frame_rate REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                codec TEXT,
                checksum TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)

        # Annotations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_width INTEGER NOT NULL,
                bbox_height INTEGER NOT NULL,
                area_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                image_path TEXT,
                annotated_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0),
                CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
            )
        """)

        # Detections table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                detector_type TEXT NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_width INTEGER NOT NULL,
                bbox_height INTEGER NOT NULL,
                confidence REAL NOT NULL,
                area_type TEXT NOT NULL,
                needs_review BOOLEAN DEFAULT FALSE,
                reviewed_by TEXT,
                reviewed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_metadata JSON,
                CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0),
                CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
            )
        """)

        # Timelines table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS timelines (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                timeline_data JSON NOT NULL,
                status TEXT DEFAULT 'draft',
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                CONSTRAINT valid_status CHECK (status IN ('draft', 'approved', 'processed', 'archived'))
            )
        """)

        # Processing jobs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                timeline_id TEXT NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
                status TEXT DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                output_path TEXT,
                error_message TEXT,
                processing_config JSON,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                CONSTRAINT valid_progress CHECK (progress >= 0.0 AND progress <= 100.0)
            )
        """)

        # Models table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                area_type TEXT NOT NULL,
                version TEXT DEFAULT '1.0',
                file_path TEXT NOT NULL,
                training_data_count INTEGER DEFAULT 0,
                accuracy_metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)

        # Schema version table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert schema version
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

        conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_filepath ON videos(filepath)",
            "CREATE INDEX IF NOT EXISTS idx_annotations_video_timestamp ON annotations(video_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_annotations_area_type ON annotations(area_type)",
            "CREATE INDEX IF NOT EXISTS idx_detections_video_timestamp ON detections(video_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_detections_needs_review ON detections(needs_review) WHERE needs_review = TRUE",
            "CREATE INDEX IF NOT EXISTS idx_timelines_video_status ON timelines(video_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_video ON processing_jobs(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_models_type_area ON models(model_type, area_type)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        conn.commit()

    # Video operations
    def save_video(self, video: VideoRecord) -> str:
        """Save a video record to the database."""
        with self._get_connection() as conn:
            data = video.model_dump()
            # Convert Path objects to strings for JSON serialization
            data["filepath"] = str(data["filepath"])
            data["created_at"] = data["created_at"].isoformat()
            data["updated_at"] = data["updated_at"].isoformat()
            data["metadata"] = json.dumps(data["metadata"])

            conn.execute(
                """
                INSERT OR REPLACE INTO videos
                (id, filename, filepath, duration, frame_rate, width, height, file_size,
                 codec, checksum, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["id"],
                    data["filename"],
                    data["filepath"],
                    data["duration"],
                    data["frame_rate"],
                    data["width"],
                    data["height"],
                    data["file_size"],
                    data["codec"],
                    data["checksum"],
                    data["created_at"],
                    data["updated_at"],
                    data["metadata"],
                ),
            )
            conn.commit()

        logger.debug("Video saved: %s", video.id)
        return video.id

    def get_video(self, video_id: str) -> VideoRecord | None:
        """Get a video record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()

            if not row:
                return None

            data = dict(row)
            data["filepath"] = Path(data["filepath"])
            data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}

            return VideoRecord(**data)

    def list_videos(self, limit: int | None = None) -> list[VideoRecord]:
        """List all video records."""
        with self._get_connection() as conn:
            query = "SELECT * FROM videos ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            results = []

            for row in cursor.fetchall():
                data = dict(row)
                data["filepath"] = Path(data["filepath"])
                data["metadata"] = (
                    json.loads(data["metadata"]) if data["metadata"] else {}
                )
                results.append(VideoRecord(**data))

            return results

    def delete_video(self, video_id: str) -> bool:
        """Delete a video record and all related data."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Video deleted: %s", video_id)

            return deleted

    # Annotation operations
    def save_annotation(self, annotation: AnnotationRecord) -> str:
        """Save an annotation record to the database."""
        with self._get_connection() as conn:
            data = annotation.model_dump()
            # Handle nested BoundingBox
            bbox = data["bounding_box"]
            data.update(
                {
                    "bbox_x": bbox["x"],
                    "bbox_y": bbox["y"],
                    "bbox_width": bbox["width"],
                    "bbox_height": bbox["height"],
                }
            )
            del data["bounding_box"]

            # Handle other fields
            data["image_path"] = str(data["image_path"]) if data["image_path"] else None
            data["created_at"] = data["created_at"].isoformat()
            data["metadata"] = json.dumps(data["metadata"])

            conn.execute(
                """
                INSERT OR REPLACE INTO annotations
                (id, video_id, timestamp, frame_number, bbox_x, bbox_y, bbox_width, bbox_height,
                 area_type, confidence, image_path, annotated_by, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["id"],
                    data["video_id"],
                    data["timestamp"],
                    data["frame_number"],
                    data["bbox_x"],
                    data["bbox_y"],
                    data["bbox_width"],
                    data["bbox_height"],
                    data["area_type"],
                    data["confidence"],
                    data["image_path"],
                    data["annotated_by"],
                    data["created_at"],
                    data["metadata"],
                ),
            )
            conn.commit()

        logger.debug("Annotation saved: %s", annotation.id)
        return annotation.id

    def get_annotations_for_video(
        self, video_id: str, area_type: str | None = None
    ) -> list[AnnotationRecord]:
        """Get all annotations for a video."""
        with self._get_connection() as conn:
            query = "SELECT * FROM annotations WHERE video_id = ?"
            params = [video_id]

            if area_type:
                query += " AND area_type = ?"
                params.append(area_type)

            query += " ORDER BY timestamp ASC"

            cursor = conn.execute(query, params)
            results = []

            for row in cursor.fetchall():
                data = dict(row)
                # Reconstruct BoundingBox
                data["bounding_box"] = {
                    "x": data.pop("bbox_x"),
                    "y": data.pop("bbox_y"),
                    "width": data.pop("bbox_width"),
                    "height": data.pop("bbox_height"),
                }
                data["image_path"] = (
                    Path(data["image_path"]) if data["image_path"] else None
                )
                data["metadata"] = (
                    json.loads(data["metadata"]) if data["metadata"] else {}
                )

                results.append(AnnotationRecord(**data))

            return results

    def get_storage_stats(self) -> dict[str, Any]:
        """Get comprehensive storage statistics."""
        if not self._initialized:
            return {"error": "Storage not initialized"}

        try:
            with self._get_connection() as conn:
                stats = {}

                # Table counts
                table_queries = {
                    "videos": "SELECT COUNT(*) FROM videos",
                    "annotations": "SELECT COUNT(*) FROM annotations",
                    "detections": "SELECT COUNT(*) FROM detections",
                    "timelines": "SELECT COUNT(*) FROM timelines",
                    "processing_jobs": "SELECT COUNT(*) FROM processing_jobs",
                    "models": "SELECT COUNT(*) FROM models",
                }
                for table, query in table_queries.items():
                    cursor = conn.execute(query)
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                # Database file size
                db_size = (
                    self.config.database_path.stat().st_size
                    if self.config.database_path.exists()
                    else 0
                )
                stats["database_size_bytes"] = db_size
                stats["database_size_mb"] = round(db_size / (1024 * 1024), 2)

                # Schema version
                cursor = conn.execute(
                    "SELECT version FROM schema_version "
                    "ORDER BY applied_at DESC LIMIT 1"
                )
                version_row = cursor.fetchone()
                stats["schema_version"] = version_row[0] if version_row else "unknown"

                return stats

        except Exception:
            msg = "Error getting storage stats"
            logger.exception(msg)
            return {"error": msg}

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(UTC).isoformat()

    def validate_integrity(self) -> ValidationResult:
        """Validate database integrity and consistency."""
        result = ValidationResult(is_valid=True)

        if not self._initialized:
            result.add_error("Storage service not initialized")
            return result

        try:
            with self._get_connection() as conn:
                # Check foreign key constraints
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()

                if fk_violations:
                    result.add_error(
                        f"Foreign key violations found: {len(fk_violations)}"
                    )
                    for violation in fk_violations:
                        result.add_error(f"FK violation in {violation[0]}: {violation}")

                # Check for orphaned records
                orphaned_annotations = conn.execute("""
                    SELECT COUNT(*) FROM annotations a
                    LEFT JOIN videos v ON a.video_id = v.id
                    WHERE v.id IS NULL
                """).fetchone()[0]

                if orphaned_annotations > 0:
                    result.add_warning(
                        f"Found {orphaned_annotations} orphaned annotations"
                    )

                # Add stats to context
                result.context.update(self.get_storage_stats())

        except Exception:
            result.add_error("Integrity check failed")
            logger.exception("Integrity check failed")

        return result
