# IMPLEMENTATION_PLAN.md - Video Privacy Editor

## Overview

This document provides a comprehensive, step-by-step implementation strategy for the Video Privacy Editor system. The plan follows a dependency-driven approach, building from foundational components to complete functionality while maintaining testability and quality at each stage.

## Implementation Strategy

### Core Philosophy
- **Bottom-Up Approach**: Build foundational layers first (core → utils → storage → detection → processing → web)
- **Incremental Development**: Each phase produces working, testable components
- **Validation-First**: Implement comprehensive validation and error handling from the start
- **Test-Driven**: Write tests alongside implementation, not as an afterthought
- **Documentation-Driven**: Update documentation as implementation progresses

### Dependency Analysis

```
Web Module
    ↓
Processing Module ← Detection Module
    ↓                 ↓
Storage Module ←------┘
    ↓
Utils Module
    ↓
Core Module (Foundation)
```

**Rationale**: This order ensures each module can be fully implemented and tested before dependent modules are built.

## Phase 1: Core Foundation Layer

**Duration**: 1-2 weeks  
**Goal**: Establish the fundamental data structures and models that all other modules depend on.

### Phase 1.1: Basic Data Models (Days 1-3)

#### Implementation Order:
1. **BoundingBox** - Most fundamental, no dependencies
2. **ValidationResult** - Used throughout for error handling
3. **SensitiveArea** - Depends on BoundingBox
4. **DetectionResult** - Depends on BoundingBox
5. **BlurRegion** - Depends on BoundingBox
6. **Timeline** - Depends on BlurRegion

#### Detailed Tasks:

**Task 1.1.1: BoundingBox Implementation**
```python
# File: mivideoeditor/core/models.py
@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    def __post_init__(self):
        # Implementation with validation
    
    @property
    def area(self) -> int:
        # Implementation
    
    def iou(self, other: 'BoundingBox') -> float:
        # Intersection over Union calculation
    
    # Additional utility methods
```

**Acceptance Criteria**:
- [ ] All validation rules implemented and tested
- [ ] Utility methods (area, center, iou, contains, overlaps) working
- [ ] JSON serialization/deserialization functional
- [ ] Comprehensive test coverage (>95%)
- [ ] Type hints complete and mypy-clean

**Task 1.1.2: ValidationResult Implementation**
```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        # Implementation
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        # Combine multiple validation results
```

**Task 1.1.3-6: Remaining Core Models**
- Implement each model following the established pattern
- Ensure proper validation in `__post_init__`
- Add utility methods as specified in the design
- Complete JSON serialization support

### Phase 1.2: Configuration Models (Days 4-5)

#### Implementation Order:
1. **DetectionConfig**
2. **ProcessingConfig** 
3. **StorageConfig**
4. **WebConfig**

#### Key Implementation Details:
```python
@dataclass
class DetectionConfig:
    frame_step: int = 10
    confidence_threshold: float = 0.7
    max_regions_per_frame: int = 5
    
    def __post_init__(self):
        # Validate all parameters
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(f"Invalid confidence threshold: {self.confidence_threshold}")
    
    @classmethod
    def from_env(cls) -> 'DetectionConfig':
        # Load from environment variables
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'DetectionConfig':
        # Load from JSON/YAML file
```

### Phase 1.3: Exception Hierarchy (Day 6)

```python
# File: mivideoeditor/core/exceptions.py
class MiVideoEditorError(Exception):
    """Base exception for all video editor errors"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class ValidationError(MiVideoEditorError):
    """Data validation errors"""
    pass

# Additional specific exceptions...
```

### Phase 1.4: Serialization System (Days 7-8)

```python
# File: mivideoeditor/core/serializers.py
class CoreJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for core models"""
    
    def default(self, obj):
        # Implementation for all core types

def serialize_core_object(obj: Any) -> Dict[str, Any]:
    """Serialize core objects to dictionary"""

def deserialize_core_object(data: Dict[str, Any], obj_type: str) -> Any:
    """Deserialize core objects from dictionary"""
```

### Phase 1.5: Constants and Validators (Days 9-10)

```python
# File: mivideoeditor/core/constants.py
SUPPORTED_AREA_TYPES = {
    'chatgpt': 'ChatGPT conversation interface',
    'atuin': 'Atuin terminal history search',
    'terminal': 'Generic terminal content',
    'custom': 'User-defined sensitive area'
}

# File: mivideoeditor/core/validators.py
def validate_bounding_box(bbox: BoundingBox, frame_size: Optional[Tuple[int, int]] = None) -> ValidationResult:
    """Comprehensive bounding box validation"""

def validate_confidence(confidence: float) -> ValidationResult:
    """Validate confidence score"""
```

**Phase 1 Deliverables:**
- [ ] Complete core models with validation
- [ ] Comprehensive test suite (>95% coverage)
- [ ] JSON serialization working
- [ ] Exception hierarchy established
- [ ] Constants and validators implemented
- [ ] Documentation updated
- [ ] Type checking passes (mypy)

---

## Phase 2: Utils Foundation Layer

**Duration**: 1-2 weeks  
**Goal**: Implement essential utilities that other modules depend on.

### Phase 2.1: System and Validation Utils (Days 1-3)

#### Priority Order:
1. **SystemUtils** - Required for dependency checking and performance estimation
2. **ValidationUtils** - Required by all other modules for data validation
3. **Decorators** - Required for caching, retry logic, and performance monitoring

**Task 2.1.1: SystemUtils Implementation**
```python
# File: mivideoeditor/utils/system.py
class SystemUtils:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        # Implementation with psutil
    
    @staticmethod
    def check_dependencies() -> Dict[str, Dict[str, Any]]:
        # Check FFmpeg, OpenCV, etc.
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        # GPU detection for hardware acceleration
```

### Phase 2.2: Time and String Utils (Days 4-6)

**Task 2.2.1: TimeUtils Implementation**
```python
# File: mivideoeditor/utils/time.py
class TimeUtils:
    @staticmethod
    def parse_time_string(time_str: str) -> float:
        # Parse HH:MM:SS.mmm, human readable, etc.
    
    @staticmethod
    def format_duration(seconds: float, format_type: str = 'hms') -> str:
        # Format for display
    
    @staticmethod
    def merge_overlapping_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Timeline optimization helper
```

### Phase 2.3: File and Math Utils (Days 7-10)

**Task 2.3.1: Video and Image Utils**
```python
# File: mivideoeditor/utils/video.py
class VideoUtils:
    @staticmethod
    def get_video_info(video_path: Path) -> VideoInfo:
        # FFprobe integration
    
    @staticmethod
    def validate_video_file(video_path: Path) -> ValidationResult:
        # Comprehensive video validation
```

**Task 2.3.2: Image Processing Utils**
```python
# File: mivideoeditor/utils/image.py
class ImageUtils:
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        # Smart resizing with aspect ratio
    
    @staticmethod
    def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        # Multiple similarity metrics
```

**Phase 2 Deliverables:**
- [ ] All utility functions implemented and tested
- [ ] System dependency checking working
- [ ] Video metadata extraction functional
- [ ] Image processing utilities complete
- [ ] Time parsing and formatting working
- [ ] Performance decorators implemented
- [ ] Comprehensive test coverage

---

## Phase 3: Storage Layer Implementation

**Duration**: 2-3 weeks  
**Goal**: Implement persistent storage with SQLite and file management.

### Phase 3.1: Database Schema and Migrations (Days 1-4)

**Task 3.1.1: Schema Implementation**
```python
# File: mivideoeditor/storage/sqlite_storage.py
class SQLiteStorage(BaseStorage):
    def create_tables(self) -> None:
        # Implement all table creation
    
    def migrate_schema(self, target_version: str) -> None:
        # Version management and migrations
```

**Task 3.1.2: Database Connection Management**
```python
class ConnectionPool:
    def __init__(self, database: Path, max_connections: int = 20):
        # Connection pooling implementation
    
    def get_connection(self) -> sqlite3.Connection:
        # Thread-safe connection retrieval
```

### Phase 3.2: File Management System (Days 5-8)

**Task 3.2.1: FileManager Implementation**
```python
# File: mivideoeditor/storage/file_manager.py
class FileManager:
    def store_video(self, source_path: Path, video_id: str) -> Path:
        # Atomic video storage with verification
    
    def store_frame_image(self, image: np.ndarray, annotation_id: str) -> Path:
        # Optimized frame storage
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        # Cleanup unreferenced files
```

### Phase 3.3: Specialized Storage Classes (Days 9-15)

**Task 3.3.1: AnnotationStorage**
```python
# File: mivideoeditor/storage/annotation_storage.py
class AnnotationStorage:
    def save_annotation(self, annotation: SensitiveArea) -> str:
        # Save with caching and validation
    
    def get_annotations_for_video(self, video_id: str) -> List[SensitiveArea]:
        # Efficient retrieval with caching
```

**Task 3.3.2: TimelineManager**
```python
# File: mivideoeditor/storage/timeline_manager.py
class TimelineManager:
    def save_timeline(self, timeline: Timeline) -> str:
        # Versioning and approval workflow
    
    def load_timeline(self, timeline_id: str) -> Optional[Timeline]:
        # Load with caching
```

### Phase 3.4: Caching and Backup (Days 16-21)

**Task 3.4.1: CacheManager**
```python
# File: mivideoeditor/storage/cache_manager.py
class CacheManager:
    def __init__(self, config: CacheConfig):
        # Multi-level caching (memory + disk)
    
    def get(self, key: str) -> Optional[Any]:
        # LRU with TTL
    
    def invalidate_pattern(self, pattern: str) -> int:
        # Pattern-based cache invalidation
```

**Phase 3 Deliverables:**
- [ ] Complete SQLite storage implementation
- [ ] File management with atomic operations
- [ ] Caching system with LRU eviction
- [ ] Backup and restore functionality
- [ ] Database migrations working
- [ ] Performance benchmarks meet requirements
- [ ] Data integrity tests passing

---

## Phase 4: Detection Layer Implementation

**Duration**: 2-3 weeks  
**Goal**: Implement detection algorithms with template matching and motion tracking.

### Phase 4.1: Base Detection Framework (Days 1-4)

**Task 4.1.1: BaseDetector Interface**
```python
# File: mivideoeditor/detection/base.py
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> DetectionResult:
        pass
    
    @abstractmethod
    def train(self, annotations: List[SensitiveArea]) -> None:
        pass
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        # Default implementation with optimization hooks
```

### Phase 4.2: Template Detector (Days 5-10)

**Task 4.2.1: Color Masking Implementation**
```python
# File: mivideoeditor/detection/template.py
class TemplateDetector(BaseDetector):
    def _apply_color_filters(self, frame: np.ndarray) -> List[BoundingBox]:
        # HSV-based color filtering for ChatGPT/Atuin
    
    def _match_templates_in_region(self, frame: np.ndarray, region: BoundingBox) -> List[Detection]:
        # Template matching within candidate regions
```

**Task 4.2.2: Template Training System**
```python
# File: mivideoeditor/detection/training.py
class TemplateTrainer:
    def train_from_annotations(self, annotations: List[SensitiveArea]) -> Dict[str, np.ndarray]:
        # Generate optimized templates from user annotations
    
    def _cluster_templates(self, templates: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        # Group similar templates
```

### Phase 4.3: Motion Tracking (Days 11-15)

**Task 4.3.1: MotionTracker Implementation**
```python
# File: mivideoeditor/detection/motion_tracker.py
class MotionTracker:
    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[TrackedDetection]:
        # Kalman filter-based tracking
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Dict[Detection, int]:
        # Hungarian algorithm for detection-to-track association
```

### Phase 4.4: Ensemble and Optimization (Days 16-21)

**Task 4.4.1: EnsembleDetector**
```python
# File: mivideoeditor/detection/ensemble.py
class EnsembleDetector(BaseDetector):
    def _voting_ensemble(self, results: List[DetectionResult]) -> DetectionResult:
        # Weighted voting with confidence scores
    
    def _cascade_ensemble(self, frame: np.ndarray) -> DetectionResult:
        # Fast detector → accurate detector pipeline
```

**Phase 4 Deliverables:**
- [ ] Complete detection framework implemented
- [ ] Template detector with color masking working
- [ ] Motion tracking functional
- [ ] Training system operational
- [ ] Ensemble methods implemented
- [ ] Performance benchmarks met
- [ ] Accuracy validation against test dataset

---

## Phase 5: Processing Layer Implementation

**Duration**: 2-3 weeks  
**Goal**: Implement video processing with FFmpeg integration and blur filters.

### Phase 5.1: Blur Filter System (Days 1-8)

**Task 5.1.1: Base Filter Interface**
```python
# File: mivideoeditor/processing/blur_filters/base.py
class BaseBlurFilter(ABC):
    @abstractmethod
    def get_ffmpeg_filter(self, region: BlurRegion, video_info: VideoInfo) -> str:
        pass
    
    @abstractmethod
    def apply_to_image(self, image: np.ndarray, region: BoundingBox) -> np.ndarray:
        pass
```

**Task 5.1.2: Individual Filter Implementations**
```python
# Implement in parallel:
# - GaussianBlur (mivideoeditor/processing/blur_filters/gaussian.py)
# - PixelateBlur (mivideoeditor/processing/blur_filters/pixelate.py)
# - NoiseOverlay (mivideoeditor/processing/blur_filters/noise.py)
# - CompositeBlur (mivideoeditor/processing/blur_filters/composite.py)
# - MotionBlur (mivideoeditor/processing/blur_filters/motion_blur.py)
```

### Phase 5.2: FFmpeg Integration (Days 9-15)

**Task 5.2.1: FFmpegWrapper**
```python
# File: mivideoeditor/processing/ffmpeg_wrapper.py
class FFmpegWrapper:
    def build_processing_command(self, input_path: Path, output_path: Path) -> List[str]:
        # Command generation with hardware acceleration
    
    def execute_with_progress(self, cmd: List[str]) -> ExecutionResult:
        # Progress monitoring and error handling
```

**Task 5.2.2: FilterChainBuilder**
```python
# File: mivideoeditor/processing/ffmpeg_wrapper.py
class FilterChainBuilder:
    def build_filter_chain(self, timeline: Timeline) -> str:
        # Generate complex filter graphs
    
    def _optimize_overlapping_regions(self, regions: List[BlurRegion]) -> List[BlurRegion]:
        # Region merging for performance
```

### Phase 5.3: Video Processor (Days 16-21)

**Task 5.3.1: Main Processing Pipeline**
```python
# File: mivideoeditor/processing/video_processor.py
class FFmpegVideoProcessor(BaseVideoProcessor):
    def process_video(self, input_path: Path, timeline: Timeline, output_path: Path) -> ProcessingResult:
        # Complete processing pipeline
    
    def _process_chunked(self, input_path: Path, timeline: Timeline) -> ProcessingResult:
        # Chunked processing for large files
```

**Phase 5 Deliverables:**
- [ ] All blur filters implemented and tested
- [ ] FFmpeg integration working with progress monitoring
- [ ] Video processing pipeline functional
- [ ] Quality profiles implemented
- [ ] Chunked processing for large videos working
- [ ] Hardware acceleration functional
- [ ] Performance benchmarks achieved

---

## Phase 6: Web Interface Implementation

**Duration**: 3-4 weeks  
**Goal**: Implement FastAPI backend and interactive frontend.

### Phase 6.1: API Foundation (Days 1-7)

**Task 6.1.1: FastAPI Application Setup**
```python
# File: mivideoeditor/web/app.py
def create_app(config: WebConfig) -> FastAPI:
    # Application factory with middleware
    
# File: mivideoeditor/web/models/
# Implement all Pydantic models for API
```

**Task 6.1.2: Core API Endpoints**
```python
# Implement in order:
# 1. mivideoeditor/web/api/system.py (health checks, status)
# 2. mivideoeditor/web/api/videos.py (upload, metadata)
# 3. mivideoeditor/web/api/annotations.py (CRUD operations)
```

### Phase 6.2: Advanced API Features (Days 8-14)

**Task 6.2.1: Processing Endpoints**
```python
# File: mivideoeditor/web/api/processing.py
@router.post("/jobs")
async def create_processing_job(job_data: ProcessingJobCreate) -> JobResponse:
    # Background job creation
    
@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobResponse:
    # Job status and progress
```

**Task 6.2.2: WebSocket Integration**
```python
# File: mivideoeditor/web/websocket/progress.py
@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    # Real-time progress updates
```

### Phase 6.3: Frontend Interface (Days 15-21)

**Task 6.3.1: Annotation Interface**
```javascript
// File: mivideoeditor/web/static/js/annotation.js
class AnnotationInterface {
    constructor(containerId, videoUrl) {
        // Interactive canvas-based annotation
    }
    
    startDrawing(e) {
        // Bounding box drawing
    }
}
```

**Task 6.3.2: Timeline Editor**
```javascript
// File: mivideoeditor/web/static/js/timeline.js
class TimelineEditor {
    renderTimeline() {
        // Visual timeline with drag-and-drop
    }
}
```

### Phase 6.4: Authentication and Security (Days 22-28)

**Task 6.4.1: Authentication System**
```python
# File: mivideoeditor/web/middleware/auth.py
class AuthManager:
    def verify_token(self, token: str) -> dict:
        # JWT token verification
    
    def create_access_token(self, data: dict) -> str:
        # Token creation
```

**Phase 6 Deliverables:**
- [ ] Complete REST API with all endpoints
- [ ] WebSocket real-time updates working
- [ ] Interactive annotation interface functional
- [ ] Timeline editor implemented
- [ ] Authentication and authorization working
- [ ] Security measures implemented
- [ ] Frontend-backend integration complete

---

## Phase 7: Integration and Testing

**Duration**: 2-3 weeks  
**Goal**: Complete system integration and comprehensive testing.

### Phase 7.1: System Integration (Days 1-7)

**Task 7.1.1: End-to-End Pipeline Testing**
```python
# File: tests/integration/test_complete_pipeline.py
def test_video_upload_to_processing():
    # Complete workflow test
    
def test_annotation_to_timeline_generation():
    # Data flow validation
```

**Task 7.1.2: Performance Integration**
- Load testing with realistic video files
- Memory usage profiling
- Processing speed validation

### Phase 7.2: User Acceptance Testing (Days 8-14)

**Task 7.2.1: User Workflow Testing**
- Annotation workflow validation
- Timeline review and adjustment
- Processing job monitoring

**Task 7.2.2: Error Handling Validation**
- Recovery from processing failures
- Invalid input handling
- System resource exhaustion scenarios

### Phase 7.3: Production Readiness (Days 15-21)

**Task 7.3.1: Deployment Preparation**
```dockerfile
# File: Dockerfile
FROM python:3.9-slim
# Production container setup
```

**Task 7.3.2: Monitoring and Logging**
```python
# File: mivideoeditor/monitoring/
# Comprehensive logging and metrics collection
```

---

## Implementation Guidelines

### Daily Development Workflow

1. **Morning**: Review previous day's work, update documentation
2. **Development**: Implement according to phase plan
3. **Testing**: Write and run tests for new functionality
4. **Code Review**: Self-review code quality and adherence to patterns
5. **Documentation**: Update relevant documentation
6. **Planning**: Prepare for next day's tasks

### Quality Gates

**Before Moving to Next Phase:**
- [ ] All tests passing (unit, integration, performance)
- [ ] Code coverage meets targets (>90% for core, >80% for web)
- [ ] Type checking passes (mypy --strict)
- [ ] Code formatting consistent (black, isort)
- [ ] Documentation updated
- [ ] Performance benchmarks met
- [ ] Security review completed

### Risk Mitigation

**Technical Risks:**
- **FFmpeg Integration Complexity**: Prototype early, maintain fallback options
- **Large Video Memory Usage**: Implement chunked processing from start
- **Detection Accuracy**: Build comprehensive test dataset early
- **Web Interface Complexity**: Start with simple HTML/JS, enhance incrementally

**Mitigation Strategies:**
- **Spike Solutions**: Prototype risky components before full implementation
- **Incremental Development**: Each phase produces working components
- **Comprehensive Testing**: Test with real video files throughout
- **Performance Monitoring**: Benchmark at each phase

### Success Metrics

**Phase Completion Criteria:**
- All planned functionality implemented
- Test coverage targets met
- Performance requirements satisfied
- Documentation current and accurate
- No critical bugs or security vulnerabilities

**Overall Project Success:**
- Process 2-4 hour videos within 2x realtime (balanced mode)
- Detect ChatGPT and Atuin interfaces with >90% accuracy
- Generate reviewable blur timelines
- Provide intuitive web interface for annotation
- Maintain video quality while ensuring privacy

---

## Alternative Implementation Approaches

### Alternative 1: Detection-First Approach
**Pros**: Earlier validation of core detection algorithms  
**Cons**: No storage layer for testing, harder to validate results  
**Verdict**: Less optimal due to testing difficulties

### Alternative 2: Web-First Approach  
**Pros**: Earlier user feedback, visible progress  
**Cons**: No backend functionality to demonstrate, unstable foundation  
**Verdict**: Risky without solid backend foundation

### Alternative 3: Parallel Development
**Pros**: Faster overall development  
**Cons**: Integration complexity, dependency conflicts  
**Verdict**: Good for larger teams, risky for solo development

## Recommended Approach: Sequential Bottom-Up

The proposed **Core → Utils → Storage → Detection → Processing → Web** approach is optimal because:

1. **Solid Foundation**: Each layer provides stable APIs for the next
2. **Testability**: Components can be thoroughly tested before use
3. **Risk Reduction**: Major architectural issues discovered early
4. **Clear Dependencies**: No circular dependencies or integration surprises
5. **Incremental Value**: Each phase produces working, valuable components

This approach ensures a stable, maintainable system built on proven foundations, with comprehensive testing and documentation at each stage.