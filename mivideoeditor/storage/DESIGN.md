# Storage Module Design

## Overview

The `storage` module handles all data persistence and management for the Video Privacy Editor system. It provides a unified interface for storing annotations, detection results, timelines, and managing file operations while ensuring data integrity, efficient retrieval, and proper cleanup.

## Design Principles

1. **Data Integrity**: ACID compliance for critical operations
2. **Performance**: Efficient queries and batch operations
3. **Scalability**: Support for large video files and datasets
4. **Flexibility**: Support multiple storage backends
5. **Security**: Secure handling of sensitive metadata
6. **Cleanup**: Automatic cleanup of temporary files
7. **Backup**: Built-in backup and recovery mechanisms

## Module Structure

```
storage/
├── __init__.py
├── DESIGN.md
├── base.py              # Abstract storage interfaces
├── sqlite_storage.py    # SQLite implementation
├── file_manager.py      # File system operations
├── annotation_storage.py # Annotation-specific operations
├── timeline_manager.py   # Timeline persistence
├── cache_manager.py     # Caching layer
├── backup_manager.py    # Backup and recovery
├── migrations/         # Database schema migrations
└── schemas/           # JSON schemas for validation
```

## Storage Architecture

### Multi-Layer Architecture

```
Application Layer
       ↓
Storage Interface Layer (Abstract APIs)
       ↓
Implementation Layer (SQLite, File System)
       ↓
Physical Storage Layer (Files, Database)
```

### Data Flow

```
User Request → Storage Interface → Implementation → Physical Storage
     ↓              ↓                   ↓              ↓
Validation → Cache Check → Database/File → Persistence
     ↓              ↓                   ↓              ↓
Response ← Cache Update ← Result Format ← Data Retrieval
```

## Core Components

### BaseStorage (Abstract Interface)

**Purpose**: Define unified interface for all storage operations.

```python
class BaseStorage(ABC):
    """Abstract base class for all storage implementations"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.connection = None
        self.cache = CacheManager(config.cache_config)
        self.file_manager = FileManager(config.data_dir)
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage backend"""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Cleanup and close connections"""
        pass
        
    @abstractmethod
    def create_tables(self) -> None:
        """Create necessary database tables/structures"""
        pass
        
    @abstractmethod
    def migrate_schema(self, target_version: str) -> None:
        """Migrate database schema to target version"""
        pass
        
    @abstractmethod
    def backup(self, backup_path: Path) -> bool:
        """Create backup of all data"""
        pass
        
    @abstractmethod
    def restore(self, backup_path: Path) -> bool:
        """Restore data from backup"""
        pass
        
    @abstractmethod
    def vacuum(self) -> None:
        """Optimize storage (vacuum, cleanup, etc.)"""
        pass
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass
        
    def validate_data_integrity(self) -> List[str]:
        """Check data integrity and return issues"""
        pass
```

### SQLiteStorage

**Purpose**: Main storage implementation using SQLite for metadata and file system for binary data.

**Design Decisions**:
- **SQLite for metadata**: ACID compliance, good performance for reads
- **File system for binary data**: Images, videos, model files
- **WAL mode**: Better concurrent access
- **Foreign key constraints**: Data integrity
- **Prepared statements**: SQL injection prevention

```python
class SQLiteStorage(BaseStorage):
    """SQLite-based storage implementation"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.db_path = config.database_path
        self.connection_pool = ConnectionPool(
            database=self.db_path,
            max_connections=config.max_connections,
            timeout=config.timeout
        )
        
    def initialize(self) -> None:
        """Initialize SQLite database"""
        # Enable WAL mode for better concurrency
        self._execute("PRAGMA journal_mode=WAL")
        self._execute("PRAGMA synchronous=NORMAL") 
        self._execute("PRAGMA cache_size=10000")
        self._execute("PRAGMA foreign_keys=ON")
        
        # Create tables if they don't exist
        self.create_tables()
        
        # Run any pending migrations
        current_version = self._get_schema_version()
        if current_version < LATEST_SCHEMA_VERSION:
            self.migrate_schema(LATEST_SCHEMA_VERSION)
    
    def create_tables(self) -> None:
        """Create all necessary database tables"""
        
        # Videos table
        self._execute("""
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
        
        # Scenes table (sensitive time ranges)
        self._execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            scene_type TEXT NOT NULL,
            description TEXT,
            priority TEXT DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            CONSTRAINT valid_time_range CHECK (end_time > start_time),
            CONSTRAINT valid_scene_type CHECK (scene_type IN ('chatgpt', 'atuin', 'terminal', 'custom')),
            CONSTRAINT valid_priority CHECK (priority IN ('low', 'medium', 'high'))
        )
        """)
        
        # Annotations table (manual bounding box annotations)
        self._execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            scene_id TEXT REFERENCES scenes(id) ON DELETE SET NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_width INTEGER NOT NULL,
            bbox_height INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            image_path TEXT,
            annotated_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON,
            CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0 AND bbox_x >= 0 AND bbox_y >= 0),
            CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
        )
        """)
        
        # Detections table (automated detection results)
        self._execute("""
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
            needs_review BOOLEAN DEFAULT FALSE,
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            detection_metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0),
            CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
        )
        """)
        
        # Timelines table (blur timeline definitions)
        self._execute("""
        CREATE TABLE IF NOT EXISTS timelines (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            name TEXT,
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
        self._execute("""
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
        
        # Models table (trained detector models)
        self._execute("""
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
        
        # Create indexes for performance
        self._create_indexes()
    
    def _create_indexes(self) -> None:
        """Create database indexes for optimal query performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_filepath ON videos(filepath)",
            "CREATE INDEX IF NOT EXISTS idx_scenes_video_time ON scenes(video_id, start_time, end_time)",
            "CREATE INDEX IF NOT EXISTS idx_annotations_video_timestamp ON annotations(video_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_annotations_scene ON annotations(scene_id)",
            "CREATE INDEX IF NOT EXISTS idx_detections_video_timestamp ON detections(video_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_detections_needs_review ON detections(needs_review) WHERE needs_review = TRUE",
            "CREATE INDEX IF NOT EXISTS idx_timelines_video_status ON timelines(video_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_video ON processing_jobs(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_models_type_area ON models(model_type, area_type)"
        ]
        
        for index_sql in indexes:
            self._execute(index_sql)
```

### FileManager

**Purpose**: Manage file system operations for binary data (images, videos, models).

**Design Features**:
- **Organized directory structure**: Videos, annotations, models, cache, temp
- **Atomic file operations**: Prevent corruption during writes
- **Automatic cleanup**: Remove orphaned and temporary files
- **Storage optimization**: Compress images, deduplicate files
- **Security**: Validate file types and sizes

```python
class FileManager:
    """Manage file system operations for the video editor"""
    
    def __init__(self, data_dir: Path, config: FileManagerConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        
        # Create directory structure
        self.directories = {
            'videos': self.data_dir / 'videos',
            'frames': self.data_dir / 'annotations' / 'frames',
            'models': self.data_dir / 'models',
            'timelines': self.data_dir / 'timelines',
            'output': self.data_dir / 'output',
            'cache': self.data_dir / 'cache',
            'temp': self.data_dir / 'temp',
            'backups': self.data_dir / 'backups'
        }
        
        self._create_directories()
        self._setup_cleanup_scheduler()
    
    def store_video(self, source_path: Path, video_id: str) -> Path:
        """Store video file with organized naming"""
        
        # Validate file
        self._validate_video_file(source_path)
        
        # Generate organized path
        timestamp = datetime.now().strftime("%Y%m")
        video_ext = source_path.suffix
        destination = self.directories['videos'] / timestamp / f"{video_id}{video_ext}"
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic copy operation
        temp_path = destination.with_suffix('.tmp')
        try:
            shutil.copy2(source_path, temp_path)
            temp_path.rename(destination)
            
            # Verify copy integrity
            if not self._verify_file_integrity(source_path, destination):
                destination.unlink()
                raise StorageError(f"File copy verification failed for {video_id}")
                
            logger.info(f"Video stored: {video_id} -> {destination}")
            return destination
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise StorageError(f"Failed to store video {video_id}: {e}")
    
    def store_frame_image(self, image: np.ndarray, annotation_id: str, 
                         timestamp: float, optimize: bool = True) -> Path:
        """Store extracted frame image with optimization"""
        
        # Organize by timestamp (YYYYMM/DD format)
        date_dir = datetime.fromtimestamp(timestamp).strftime("%Y%m/%d")
        frame_dir = self.directories['frames'] / date_dir
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        frame_filename = f"frame_{annotation_id}_{int(timestamp*1000):010d}.png"
        frame_path = frame_dir / frame_filename
        
        # Optimize image before saving
        if optimize:
            image = self._optimize_image(image)
        
        # Save with error handling
        try:
            cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            logger.debug(f"Frame image stored: {frame_path}")
            return frame_path
        except Exception as e:
            raise StorageError(f"Failed to store frame image {annotation_id}: {e}")
    
    def store_model(self, model_data: bytes, model_id: str, 
                   model_type: str, version: str = "1.0") -> Path:
        """Store trained model file"""
        
        # Organize by type and version
        model_dir = self.directories['models'] / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_id}.pkl"
        model_path = model_dir / model_filename
        
        # Atomic write operation
        temp_path = model_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'wb') as f:
                f.write(model_data)
            temp_path.rename(model_path)
            
            logger.info(f"Model stored: {model_id} -> {model_path}")
            return model_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise StorageError(f"Failed to store model {model_id}: {e}")
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """Clean up files not referenced in database"""
        
        # Get all file references from database
        storage = get_storage_instance()
        referenced_files = storage.get_all_file_references()
        
        cleanup_stats = {
            'frames_removed': 0,
            'models_removed': 0,
            'temp_removed': 0,
            'bytes_freed': 0
        }
        
        # Check frame images
        for frame_file in self.directories['frames'].rglob('*.png'):
            if str(frame_file) not in referenced_files:
                file_size = frame_file.stat().st_size
                frame_file.unlink()
                cleanup_stats['frames_removed'] += 1
                cleanup_stats['bytes_freed'] += file_size
        
        # Check model files
        for model_file in self.directories['models'].rglob('*.pkl'):
            if str(model_file) not in referenced_files:
                file_size = model_file.stat().st_size
                model_file.unlink()
                cleanup_stats['models_removed'] += 1
                cleanup_stats['bytes_freed'] += file_size
        
        # Clean temporary files older than 24 hours
        cutoff_time = time.time() - (24 * 3600)
        for temp_file in self.directories['temp'].rglob('*'):
            if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                file_size = temp_file.stat().st_size
                temp_file.unlink()
                cleanup_stats['temp_removed'] += 1
                cleanup_stats['bytes_freed'] += file_size
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def _optimize_image(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for storage (resize, compress)"""
        
        height, width = image.shape[:2]
        
        # Resize if too large (keep aspect ratio)
        max_dimension = self.config.max_image_dimension
        if max(height, width) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Apply compression if image is still large
        if image.nbytes > self.config.max_image_bytes:
            # Reduce quality by converting to JPEG and back to reduce size
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, encoded_img = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return image
```

### AnnotationStorage

**Purpose**: Specialized storage operations for annotation data.

```python
class AnnotationStorage:
    """Specialized storage for annotation operations"""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.file_manager = storage.file_manager
        self.cache = storage.cache
        
    def save_annotation(self, annotation: SensitiveArea, 
                       frame_image: Optional[np.ndarray] = None) -> str:
        """Save annotation with optional frame image"""
        
        # Store frame image if provided
        image_path = None
        if frame_image is not None:
            image_path = self.file_manager.store_frame_image(
                frame_image, annotation.id, annotation.timestamp
            )
        
        # Prepare database record
        record_data = {
            'id': annotation.id,
            'video_id': annotation.metadata.get('video_id'),
            'scene_id': annotation.metadata.get('scene_id'),
            'timestamp': annotation.timestamp,
            'frame_number': annotation.metadata.get('frame_number', 0),
            'bbox_x': annotation.bounding_box.x,
            'bbox_y': annotation.bounding_box.y,
            'bbox_width': annotation.bounding_box.width,
            'bbox_height': annotation.bounding_box.height,
            'label': annotation.area_type,
            'confidence': annotation.confidence,
            'image_path': str(image_path) if image_path else None,
            'annotated_by': annotation.metadata.get('annotated_by'),
            'metadata': json.dumps(annotation.metadata)
        }
        
        # Insert into database
        self.storage.execute_query(
            """INSERT OR REPLACE INTO annotations 
               (id, video_id, scene_id, timestamp, frame_number, bbox_x, bbox_y, 
                bbox_width, bbox_height, label, confidence, image_path, 
                annotated_by, metadata)
               VALUES (:id, :video_id, :scene_id, :timestamp, :frame_number,
                       :bbox_x, :bbox_y, :bbox_width, :bbox_height, :label,
                       :confidence, :image_path, :annotated_by, :metadata)""",
            record_data
        )
        
        # Invalidate relevant caches
        self.cache.invalidate_pattern(f"annotations:video:{annotation.metadata.get('video_id')}:*")
        
        logger.debug(f"Annotation saved: {annotation.id}")
        return annotation.id
    
    def load_annotation(self, annotation_id: str) -> Optional[SensitiveArea]:
        """Load annotation by ID"""
        
        # Check cache first
        cache_key = f"annotation:{annotation_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return self._deserialize_annotation(cached)
        
        # Query database
        result = self.storage.execute_query(
            "SELECT * FROM annotations WHERE id = ?",
            (annotation_id,)
        ).fetchone()
        
        if not result:
            return None
        
        annotation = self._deserialize_annotation(dict(result))
        
        # Cache result
        self.cache.set(cache_key, annotation, ttl=3600)  # 1 hour TTL
        
        return annotation
    
    def get_annotations_for_video(self, video_id: str, 
                                 area_type: Optional[str] = None) -> List[SensitiveArea]:
        """Get all annotations for a video, optionally filtered by type"""
        
        # Build cache key
        cache_key = f"annotations:video:{video_id}"
        if area_type:
            cache_key += f":type:{area_type}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return [self._deserialize_annotation(ann) for ann in cached]
        
        # Build query
        query = "SELECT * FROM annotations WHERE video_id = ?"
        params = [video_id]
        
        if area_type:
            query += " AND label = ?"
            params.append(area_type)
        
        query += " ORDER BY timestamp ASC"
        
        # Execute query
        results = self.storage.execute_query(query, params).fetchall()
        
        annotations = [self._deserialize_annotation(dict(row)) for row in results]
        
        # Cache results
        self.cache.set(cache_key, annotations, ttl=1800)  # 30 minutes TTL
        
        return annotations
    
    def get_annotations_in_time_range(self, video_id: str, start_time: float, 
                                    end_time: float) -> List[SensitiveArea]:
        """Get annotations within a time range"""
        
        results = self.storage.execute_query(
            """SELECT * FROM annotations 
               WHERE video_id = ? AND timestamp BETWEEN ? AND ?
               ORDER BY timestamp ASC""",
            (video_id, start_time, end_time)
        ).fetchall()
        
        return [self._deserialize_annotation(dict(row)) for row in results]
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation and associated files"""
        
        # Get annotation to find associated files
        annotation = self.load_annotation(annotation_id)
        if not annotation:
            return False
        
        # Delete associated frame image
        if annotation.image_path and annotation.image_path.exists():
            try:
                annotation.image_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete frame image {annotation.image_path}: {e}")
        
        # Delete from database
        self.storage.execute_query(
            "DELETE FROM annotations WHERE id = ?",
            (annotation_id,)
        )
        
        # Invalidate caches
        self.cache.invalidate_pattern(f"annotation:{annotation_id}")
        if annotation.metadata.get('video_id'):
            video_id = annotation.metadata['video_id']
            self.cache.invalidate_pattern(f"annotations:video:{video_id}:*")
        
        logger.info(f"Annotation deleted: {annotation_id}")
        return True
    
    def _deserialize_annotation(self, record: Dict[str, Any]) -> SensitiveArea:
        """Convert database record to SensitiveArea object"""
        
        # Parse metadata
        metadata = {}
        if record.get('metadata'):
            try:
                metadata = json.loads(record['metadata'])
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON for annotation {record['id']}")
        
        # Add database fields to metadata
        metadata.update({
            'video_id': record.get('video_id'),
            'scene_id': record.get('scene_id'),
            'frame_number': record.get('frame_number'),
            'annotated_by': record.get('annotated_by')
        })
        
        return SensitiveArea(
            id=record['id'],
            timestamp=record['timestamp'],
            bounding_box=BoundingBox(
                x=record['bbox_x'],
                y=record['bbox_y'], 
                width=record['bbox_width'],
                height=record['bbox_height']
            ),
            area_type=record['label'],
            confidence=record.get('confidence', 1.0),
            image_path=Path(record['image_path']) if record.get('image_path') else None,
            metadata=metadata
        )
```

### TimelineManager

**Purpose**: Manage timeline persistence and versioning.

```python
class TimelineManager:
    """Manage timeline storage and versioning"""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.cache = storage.cache
        
    def save_timeline(self, timeline: Timeline) -> str:
        """Save timeline with versioning support"""
        
        # Check if timeline already exists
        existing = self._get_existing_timeline(timeline.video_path, timeline.id)
        
        if existing:
            # Increment version
            new_version = existing['version'] + 1
            # Archive old version
            self._archive_timeline_version(existing['id'], existing['version'])
        else:
            new_version = 1
        
        # Serialize timeline data
        timeline_data = self._serialize_timeline(timeline)
        
        # Save to database
        record_data = {
            'id': timeline.id,
            'video_id': self._get_video_id_from_path(timeline.video_path),
            'name': timeline.metadata.get('name', f"Timeline v{new_version}"),
            'version': new_version,
            'timeline_data': json.dumps(timeline_data),
            'status': 'draft',
            'created_by': timeline.metadata.get('created_by'),
            'metadata': json.dumps(timeline.metadata)
        }
        
        self.storage.execute_query(
            """INSERT OR REPLACE INTO timelines
               (id, video_id, name, version, timeline_data, status, created_by, metadata)
               VALUES (:id, :video_id, :name, :version, :timeline_data, 
                       :status, :created_by, :metadata)""",
            record_data
        )
        
        # Invalidate caches
        video_id = record_data['video_id']
        self.cache.invalidate_pattern(f"timeline:*:{video_id}:*")
        
        logger.info(f"Timeline saved: {timeline.id} v{new_version}")
        return timeline.id
    
    def load_timeline(self, timeline_id: str, version: Optional[int] = None) -> Optional[Timeline]:
        """Load timeline by ID, optionally specific version"""
        
        # Build cache key
        cache_key = f"timeline:{timeline_id}"
        if version:
            cache_key += f":v{version}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return self._deserialize_timeline(cached)
        
        # Build query
        query = "SELECT * FROM timelines WHERE id = ?"
        params = [timeline_id]
        
        if version:
            query += " AND version = ?"
            params.append(version)
        else:
            query += " ORDER BY version DESC LIMIT 1"  # Get latest version
        
        # Execute query
        result = self.storage.execute_query(query, params).fetchone()
        
        if not result:
            return None
        
        timeline = self._deserialize_timeline(dict(result))
        
        # Cache result
        self.cache.set(cache_key, timeline, ttl=1800)
        
        return timeline
    
    def get_timeline_versions(self, timeline_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a timeline"""
        
        results = self.storage.execute_query(
            """SELECT id, version, status, created_at, updated_at, created_by
               FROM timelines WHERE id = ? ORDER BY version DESC""",
            (timeline_id,)
        ).fetchall()
        
        return [dict(row) for row in results]
    
    def approve_timeline(self, timeline_id: str, approved_by: str) -> bool:
        """Mark timeline as approved for processing"""
        
        result = self.storage.execute_query(
            """UPDATE timelines SET status = 'approved', updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND status = 'draft'""",
            (timeline_id,)
        )
        
        if result.rowcount > 0:
            # Invalidate caches
            self.cache.invalidate_pattern(f"timeline:{timeline_id}*")
            logger.info(f"Timeline approved: {timeline_id} by {approved_by}")
            return True
        
        return False
    
    def _serialize_timeline(self, timeline: Timeline) -> Dict[str, Any]:
        """Convert Timeline object to serializable dict"""
        
        return {
            'video_path': str(timeline.video_path),
            'video_duration': timeline.video_duration,
            'frame_rate': timeline.frame_rate,
            'blur_regions': [
                {
                    'id': region.id,
                    'start_time': region.start_time,
                    'end_time': region.end_time,
                    'bounding_box': {
                        'x': region.bounding_box.x,
                        'y': region.bounding_box.y,
                        'width': region.bounding_box.width,
                        'height': region.bounding_box.height
                    },
                    'blur_type': region.blur_type,
                    'blur_strength': region.blur_strength,
                    'interpolation': region.interpolation,
                    'confidence': region.confidence,
                    'needs_review': region.needs_review,
                    'metadata': region.metadata
                }
                for region in timeline.blur_regions
            ],
            'version': timeline.version,
            'created_at': timeline.created_at.isoformat(),
            'metadata': timeline.metadata
        }
```

### CacheManager

**Purpose**: Implement intelligent caching to improve performance.

```python
class CacheManager:
    """Multi-level caching system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = {}  # In-memory LRU cache
        self.memory_usage = 0
        self.access_times = {}  # For LRU implementation
        self.disk_cache_dir = config.disk_cache_dir
        self._setup_disk_cache()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)"""
        
        # Check memory cache first
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            return self.memory_cache[key]
        
        # Check disk cache
        if self.config.use_disk_cache:
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                # Promote to memory cache if space available
                if self._can_fit_in_memory(disk_value):
                    self.set(key, disk_value, promote_only=True)
                return disk_value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, promote_only: bool = False) -> None:
        """Set value in cache"""
        
        value_size = self._estimate_size(value)
        
        # Memory cache
        if self._can_fit_in_memory(value):
            # Make room if necessary
            while (self.memory_usage + value_size) > self.config.max_memory_bytes:
                self._evict_lru()
            
            self.memory_cache[key] = value
            self.access_times[key] = time.time()
            self.memory_usage += value_size
        
        # Disk cache (if not just promoting)
        if self.config.use_disk_cache and not promote_only:
            self._set_to_disk(key, value, ttl)
    
    def invalidate(self, key: str) -> None:
        """Invalidate specific cache entry"""
        
        # Remove from memory
        if key in self.memory_cache:
            self.memory_usage -= self._estimate_size(self.memory_cache[key])
            del self.memory_cache[key]
            del self.access_times[key]
        
        # Remove from disk
        if self.config.use_disk_cache:
            self._remove_from_disk(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching pattern"""
        
        import re
        # Convert shell-style pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex = re.compile(regex_pattern)
        
        invalidated_count = 0
        
        # Memory cache
        keys_to_remove = [key for key in self.memory_cache.keys() if regex.match(key)]
        for key in keys_to_remove:
            self.invalidate(key)
            invalidated_count += 1
        
        # Disk cache
        if self.config.use_disk_cache:
            disk_invalidated = self._invalidate_disk_pattern(regex)
            invalidated_count += disk_invalidated
        
        return invalidated_count
```

### BackupManager

**Purpose**: Handle backup and recovery operations.

```python
class BackupManager:
    """Handle backup and recovery of all data"""
    
    def __init__(self, storage: BaseStorage, config: BackupConfig):
        self.storage = storage
        self.config = config
        self.backup_dir = config.backup_directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_full_backup(self, backup_name: Optional[str] = None) -> Path:
        """Create complete backup of all data"""
        
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"full_backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup database
            db_backup_path = backup_path / "database.db"
            self._backup_database(db_backup_path)
            
            # Backup file system data
            self._backup_files(backup_path)
            
            # Create manifest
            manifest = self._create_backup_manifest(backup_path)
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Compress backup if configured
            if self.config.compress_backups:
                compressed_path = self._compress_backup(backup_path)
                shutil.rmtree(backup_path)
                backup_path = compressed_path
            
            logger.info(f"Full backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise StorageError(f"Backup failed: {e}")
    
    def restore_from_backup(self, backup_path: Path) -> bool:
        """Restore system from backup"""
        
        if not backup_path.exists():
            raise StorageError(f"Backup not found: {backup_path}")
        
        # Extract if compressed
        if backup_path.suffix in ['.zip', '.tar.gz']:
            extracted_path = self._extract_backup(backup_path)
        else:
            extracted_path = backup_path
        
        try:
            # Verify backup integrity
            if not self._verify_backup_integrity(extracted_path):
                raise StorageError("Backup integrity check failed")
            
            # Create temporary backup of current data
            temp_backup = self.create_full_backup("pre_restore_backup")
            
            # Restore database
            db_backup = extracted_path / "database.db"
            self._restore_database(db_backup)
            
            # Restore files
            self._restore_files(extracted_path)
            
            logger.info(f"System restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            # Attempt to restore from temp backup
            try:
                self.restore_from_backup(temp_backup)
                logger.info("Restored to pre-restore state")
            except Exception as restore_error:
                logger.critical(f"Failed to restore to pre-restore state: {restore_error}")
            return False
        
        finally:
            # Cleanup
            if extracted_path != backup_path and extracted_path.exists():
                shutil.rmtree(extracted_path)
```

## Configuration System

### StorageConfig

```python
@dataclass
class StorageConfig:
    """Configuration for storage system"""
    
    # Database settings
    database_path: Path = Path("data/app.db")
    max_connections: int = 20
    timeout: float = 30.0
    
    # File system settings
    data_dir: Path = Path("data")
    max_file_size: int = 10 * 1024**3  # 10GB
    
    # Cache settings
    cache_config: CacheConfig = field(default_factory=lambda: CacheConfig())
    
    # Backup settings
    backup_config: BackupConfig = field(default_factory=lambda: BackupConfig())
    
    # Performance settings
    bulk_insert_batch_size: int = 1000
    vacuum_interval_hours: int = 24
    cleanup_interval_hours: int = 6

@dataclass
class CacheConfig:
    """Cache configuration"""
    max_memory_bytes: int = 512 * 1024**2  # 512MB
    use_disk_cache: bool = True
    disk_cache_dir: Path = Path("data/cache")
    max_disk_cache_gb: float = 2.0
    default_ttl: int = 3600  # 1 hour

@dataclass  
class BackupConfig:
    """Backup configuration"""
    backup_directory: Path = Path("data/backups")
    compress_backups: bool = True
    max_backups: int = 10
    auto_backup_interval_hours: int = 24
```

## Security and Privacy

### Data Protection

- **Encryption at rest**: Optional encryption for sensitive data
- **Access controls**: Role-based permissions
- **Audit logging**: Track data access and modifications
- **Data anonymization**: Remove identifying information from exports

### Secure Deletion

```python
def secure_delete_file(file_path: Path) -> bool:
    """Securely delete file to prevent recovery"""
    
    if not file_path.exists():
        return True
    
    try:
        # Get file size
        file_size = file_path.stat().st_size
        
        # Overwrite with random data multiple times
        with open(file_path, 'r+b') as f:
            for _ in range(3):  # 3 passes
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Finally delete the file
        file_path.unlink()
        return True
        
    except Exception as e:
        logger.error(f"Secure deletion failed for {file_path}: {e}")
        return False
```

This storage design provides a robust, scalable, and secure foundation for persisting all data in the Video Privacy Editor system while maintaining performance and data integrity.