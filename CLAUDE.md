# CLAUDE.md - Video Privacy Editor Development Guide

## Project Overview

The **Video Privacy Editor** is a sophisticated system for detecting and blurring sensitive information in screen recordings. This document provides comprehensive guidance for Claude AI assistants working on this project, ensuring consistency, quality, and alignment with the established architecture.

## System Architecture Summary

### Core Philosophy
- **Privacy-First Design**: Destructive privacy protection that makes sensitive information unrecoverable
- **Modular Architecture**: Clean separation of concerns across distinct modules
- **User-Reviewable Pipeline**: Every automated decision can be reviewed and overridden
- **Performance Optimized**: Efficient processing of 2-4 hour videos with configurable quality levels
- **Extensible Framework**: Easy addition of new detectors, blur filters, and storage backends

### Module Structure
```
mivideoeditor/
├── core/          # Fundamental data structures and models
├── detection/     # Detection algorithms and motion tracking
├── storage/       # Data persistence and file management  
├── processing/    # Video processing and blur filters
├── utils/         # Common utilities and helper functions
└── web/           # Web interface and REST API
```

## Key Design Decisions & Rationale

### 1. **Detection Strategy: Template Matching + Color Masking**
- **Why**: Optimal balance between speed and accuracy for UI detection
- **Implementation**: Multi-stage pipeline (color pre-filtering → template matching → post-processing)
- **Benefits**: 10-100x speed improvement over full-frame analysis while maintaining high accuracy

### 2. **Storage: Hybrid SQLite + File System**
- **Why**: SQLite for ACID compliance on metadata, file system for binary data
- **Implementation**: SQLite with WAL mode, organized file structure, automatic cleanup
- **Benefits**: Data integrity, performance, and efficient handling of large video files

### 3. **Processing: FFmpeg Integration with Custom Filters**
- **Why**: Leverage proven video processing while adding custom privacy protection
- **Implementation**: Complex filter graph generation, hardware acceleration, chunked processing
- **Benefits**: Production-ready video processing with specialized privacy features

### 4. **Blur Strategy: Composite Destructive Filtering**
- **Why**: Ensure sensitive information cannot be reconstructed
- **Implementation**: Gaussian blur + pixelation + noise overlay
- **Benefits**: Maximum privacy protection while maintaining acceptable quality

### 5. **Web Interface: FastAPI + Interactive Canvas**
- **Why**: Modern API with real-time annotation capabilities
- **Implementation**: RESTful API, WebSocket progress updates, HTML5 canvas annotation
- **Benefits**: Intuitive user experience with real-time feedback

## Development Guidelines

### Code Quality Standards

1. **Type Safety**: Use comprehensive type hints and validation
```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, validator

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    def __post_init__(self):
        assert self.width > 0 and self.height > 0
```

2. **Error Handling**: Implement graceful degradation and recovery
```python
class ProcessingError(Exception):
    def __init__(self, message: str, error_code: str, recoverable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable
```

3. **Logging**: Comprehensive logging with appropriate levels
```python
import logging
logger = logging.getLogger(__name__)

def detect_regions(frame: np.ndarray) -> List[Detection]:
    logger.debug(f"Processing frame {frame.shape}")
    try:
        # Detection logic
        logger.info(f"Found {len(detections)} regions")
        return detections
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise
```

4. **Configuration Management**: Centralized, validated configuration
```python
@dataclass
class DetectionConfig:
    frame_step: int = 10
    confidence_threshold: float = 0.7
    max_regions_per_frame: int = 5
    
    def __post_init__(self):
        assert 0.0 < self.confidence_threshold <= 1.0
```

### Performance Considerations

1. **Memory Management**: Process large videos efficiently
- Use chunked processing for videos > 1GB
- Implement memory monitoring and adaptive chunk sizing
- Clean up temporary files promptly

2. **Caching Strategy**: Multi-level caching for performance
- Memory cache for frequently accessed data (LRU eviction)
- Disk cache for processed frames and detection results
- Database query result caching with TTL

3. **Batch Operations**: Optimize database and API operations
- Batch database inserts/updates where possible
- Use connection pooling for database access
- Implement background task queuing for long operations

### Security Requirements

1. **Input Validation**: Validate all external input rigorously
```python
def validate_video_upload(file: UploadFile) -> ValidationResult:
    if not file.content_type.startswith('video/'):
        return ValidationResult(False, ["Invalid file type"])
    
    if file.size > MAX_UPLOAD_SIZE:
        return ValidationResult(False, ["File too large"])
    
    return ValidationResult(True, [], [])
```

2. **Secure File Handling**: Prevent directory traversal and malicious uploads
```python
def sanitize_filepath(filename: str) -> str:
    # Remove directory traversal attempts
    filename = os.path.basename(filename)
    # Remove invalid characters
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
```

3. **Data Privacy**: Secure handling of sensitive video content
- Never log sensitive frame content
- Implement secure deletion for temporary files
- Optional encryption for stored annotations

## Implementation Priorities

### Phase 1: Core Foundation (Completed)
✅ **Documentation**: Comprehensive design documents for all modules  
✅ **Architecture**: Module structure and interfaces defined  
✅ **Data Models**: Core data structures and validation  

### Phase 2: Core Implementation (Next Steps)
🔄 **Data Models**: Implement core models with validation  
🔄 **Storage Layer**: SQLite storage with file management  
🔄 **Basic Detection**: Template detector with color masking  
🔄 **Simple Processing**: Single blur filter with FFmpeg  

### Phase 3: Detection & Processing
- **Advanced Detection**: Motion tracking, ensemble methods
- **Blur Filters**: All filter types (gaussian, pixelate, noise, composite)
- **Quality Management**: Adaptive quality profiles
- **Performance Optimization**: Chunked processing, hardware acceleration

### Phase 4: Web Interface
- **API Implementation**: All REST endpoints with validation
- **Frontend Interface**: Annotation and review tools
- **Real-time Updates**: WebSocket progress monitoring
- **Authentication**: User management and security

### Phase 5: Production Features
- **Batch Processing**: Multiple video handling
- **Monitoring**: Performance metrics and health checks
- **Deployment**: Docker containers and production config
- **Testing**: Comprehensive test suite

## Specific Implementation Guidance

### When Implementing Core Models

**Key Requirements**:
- All models must be JSON serializable
- Implement comprehensive validation in `__post_init__`
- Use immutable data structures where possible
- Include utility methods for common operations

**Example Pattern**:
```python
@dataclass(frozen=True)  # Immutable
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BoundingBox':
        return cls(**data)
```

### When Implementing Detectors

**Key Requirements**:
- Inherit from `BaseDetector` and implement all abstract methods
- Include confidence scoring for all detections
- Support batch processing for performance
- Implement graceful fallback strategies

**Template Pattern**:
```python
class CustomDetector(BaseDetector):
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.is_trained = False
        
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        if not self.is_trained:
            raise DetectionError("Detector not trained", "NOT_TRAINED")
        
        start_time = time.time()
        try:
            # Detection implementation
            regions = self._detect_regions(frame)
            confidences = self._calculate_confidences(regions, frame)
            
            return DetectionResult(
                regions=regions,
                confidences=confidences,
                detection_time=time.time() - start_time,
                detector_type=self.__class__.__name__,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult.empty(timestamp)
```

### When Implementing Storage Operations

**Key Requirements**:
- Use transactions for data consistency
- Implement proper connection management
- Include caching for frequently accessed data
- Handle database migrations gracefully

**Pattern for CRUD Operations**:
```python
def save_annotation(self, annotation: SensitiveArea) -> str:
    with self.storage.transaction():
        # Validate data
        validation_result = ValidationUtils.validate_annotation(annotation)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)
        
        # Check for duplicates
        existing = self.storage.get_annotation_by_id(annotation.id)
        
        # Save to database
        self.storage.execute_query(INSERT_ANNOTATION_SQL, annotation.to_dict())
        
        # Invalidate relevant caches
        self.cache.invalidate_pattern(f"annotations:video:{annotation.video_id}:*")
        
        logger.info(f"Annotation saved: {annotation.id}")
        return annotation.id
```

### When Implementing Blur Filters

**Key Requirements**:
- Inherit from `BaseBlurFilter` with both FFmpeg and OpenCV implementations
- Support configurable strength parameters
- Include performance impact estimation
- Implement preview capabilities

**Filter Pattern**:
```python
class CustomBlur(BaseBlurFilter):
    def __init__(self, param1: float = 1.0, param2: int = 10):
        config = {'param1': param1, 'param2': param2}
        super().__init__(config)
    
    def get_ffmpeg_filter(self, region: BlurRegion, video_info: VideoInfo) -> str:
        strength = int(self.config['param1'] * region.blur_strength)
        return f"custom_filter=strength={strength}"
    
    def apply_to_image(self, image: np.ndarray, region: BoundingBox) -> np.ndarray:
        # Extract region
        roi = image[region.y:region.y+region.height, region.x:region.x+region.width]
        
        # Apply custom blur logic
        blurred_roi = self._apply_custom_blur(roi)
        
        # Replace in original image
        image[region.y:region.y+region.height, region.x:region.x+region.width] = blurred_roi
        return image
```

## Testing Strategy

### Unit Tests
- **Coverage Target**: >90% for core modules, >80% for web module
- **Mock External Dependencies**: FFmpeg, file system operations
- **Test Data**: Use synthetic data and small test videos
- **Performance Tests**: Ensure operations meet timing requirements

### Integration Tests
- **End-to-End Workflows**: Complete pipeline from video to output
- **API Testing**: All endpoints with various scenarios
- **Database Testing**: Test migrations and data consistency
- **Error Scenarios**: Test failure modes and recovery

### Performance Tests
- **Load Testing**: Process multiple videos simultaneously
- **Memory Testing**: Monitor memory usage during long operations
- **Scalability Testing**: Test with various video sizes and lengths

## Common Pitfalls to Avoid

1. **Memory Leaks**: Always clean up OpenCV matrices and temporary files
2. **Blocking Operations**: Use async/await for I/O operations in web interface
3. **Race Conditions**: Proper locking for shared resources
4. **Error Swallowing**: Always log errors before handling them
5. **Hardcoded Paths**: Use configuration and Path objects
6. **SQL Injection**: Use parameterized queries exclusively
7. **Large Video Loading**: Always process videos in chunks
8. **Missing Validation**: Validate all input data thoroughly

## Code Review Checklist

### Before Submitting Code
- [ ] All type hints are present and accurate
- [ ] Error handling is comprehensive with appropriate logging
- [ ] Unit tests cover new functionality
- [ ] Configuration is externalized (no hardcoded values)
- [ ] Documentation is updated for public APIs
- [ ] Memory usage is reasonable for large inputs
- [ ] Security considerations are addressed
- [ ] Code follows established patterns in the module

### Code Review Focus Areas
- [ ] **Correctness**: Does it solve the intended problem?
- [ ] **Performance**: Will it handle expected load efficiently?
- [ ] **Security**: Are there any security vulnerabilities?
- [ ] **Maintainability**: Is the code clear and well-structured?
- [ ] **Testability**: Can the code be easily tested?
- [ ] **Error Handling**: Are edge cases and failures handled properly?

## Useful Commands and Patterns

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=mivideoeditor

# Type checking
mypy mivideoeditor/

# Code formatting
black mivideoeditor/
isort mivideoeditor/

# Start development server
uvicorn mivideoeditor.web.app:app --reload --port 8000
```

### Common Code Patterns

#### Configuration Loading
```python
@dataclass
class ModuleConfig:
    param1: str
    param2: int = 10
    
    @classmethod
    def from_env(cls) -> 'ModuleConfig':
        return cls(
            param1=os.getenv('PARAM1', 'default'),
            param2=int(os.getenv('PARAM2', '10'))
        )
```

#### Error Handling with Context
```python
def process_with_context(operation_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.info(f"Completed {operation_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation_name}: {e}")
                raise ProcessingError(f"{operation_name} failed", str(e))
        return wrapper
    return decorator
```

#### Resource Management
```python
from contextlib import contextmanager

@contextmanager
def video_processing_context(video_path: Path):
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
```

## Project-Specific Context

### Problem Domain
- **Primary Use Case**: Privacy protection for screen recordings containing ChatGPT conversations and terminal history
- **Target Users**: Content creators, developers, professionals sharing screen recordings
- **Quality Requirements**: Maintain video quality while ensuring complete privacy
- **Scale Requirements**: Handle 2-4 hour videos efficiently

### Technical Constraints
- **Platform**: Primary macOS, secondary Linux/Windows
- **Dependencies**: FFmpeg (required), OpenCV, SQLite
- **Performance**: Process videos at 0.5-2x realtime depending on quality mode
- **Storage**: Efficient handling of multi-GB video files

### Privacy Requirements
- **Destructive Protection**: Make sensitive information unrecoverable
- **User Control**: Manual review and adjustment of all automated decisions
- **No Data Leakage**: Never log or store sensitive frame content
- **Secure Processing**: Temporary files are securely deleted

## Questions for Clarification

When working on this project, always consider:

1. **Does this implementation maintain privacy-first principles?**
2. **Is this approach scalable to 4-hour videos?**
3. **Can the user review and override this automated decision?**
4. **Does this handle edge cases gracefully with proper fallbacks?**
5. **Is the performance impact reasonable for the benefit provided?**
6. **Does this maintain consistency with established patterns?**

## Future Enhancements Roadmap

### Planned Features
- **Real-time Processing**: Live blur during screen recording
- **Cloud Processing**: Scalable cloud-based video processing
- **Mobile Support**: Annotation interface for tablets
- **Advanced ML**: Deep learning-based detection models
- **Collaboration**: Multi-user annotation workflows
- **Analytics**: Processing performance analytics and optimization

### Architecture Preparation
- **Plugin System**: Dynamic loading of custom detectors and filters
- **Microservices**: Split processing into separate services
- **Event-Driven**: Pub/sub architecture for scalability
- **Containerization**: Docker and Kubernetes deployment
- **Monitoring**: Comprehensive observability stack

Remember: This is a sophisticated system with complex requirements. Always prioritize correctness and privacy over performance optimizations, and maintain the modular architecture that allows for independent development and testing of components.

---

*This document should be updated as the project evolves. When making significant architectural changes, update this guide to reflect new patterns and decisions.*