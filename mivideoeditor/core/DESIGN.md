# Core Module Design

## Overview

The `core` module contains the fundamental data structures and models that serve as the foundation for the entire Video Privacy Editor system. This module defines the contracts and interfaces that all other modules depend on.

## Design Principles

1. **Immutability Where Possible**: Core data structures should be immutable to prevent accidental modifications
2. **Type Safety**: Extensive use of type hints and validation
3. **Serializable**: All models must be JSON serializable for storage and API transport
4. **Extensible**: Models should support metadata and future extensions
5. **Validation**: Built-in validation for all data constraints

## Module Structure

```
core/
├── __init__.py
├── DESIGN.md
├── models.py          # Core data models
├── exceptions.py      # Custom exceptions
├── validators.py      # Data validation utilities
├── serializers.py     # JSON serialization helpers
└── constants.py       # System-wide constants
```

## Data Models

### 1. BoundingBox

**Purpose**: Represents a rectangular region in pixel coordinates.

**Design Decisions**:
- Use integer coordinates for pixel precision
- Validate positive dimensions
- Provide utility methods for common operations (area, center, IoU)
- Immutable after creation

```python
@dataclass(frozen=True)
class BoundingBox:
    x: int          # Left coordinate (0-based)
    y: int          # Top coordinate (0-based)  
    width: int      # Box width (must be > 0)
    height: int     # Box height (must be > 0)
```

**Key Methods**:
- `area() -> int`: Calculate area
- `center() -> Tuple[int, int]`: Get center coordinates
- `iou(other: BoundingBox) -> float`: Intersection over Union
- `contains(point: Tuple[int, int]) -> bool`: Point containment
- `overlaps(other: BoundingBox) -> bool`: Check overlap
- `to_dict() -> dict`: Serialization

### 2. SensitiveArea

**Purpose**: Represents an annotated sensitive region with metadata.

**Design Decisions**:
- Include confidence score for quality tracking
- Support flexible metadata for future extensions
- Link to source image for verification
- Unique identifier for database relations

```python
@dataclass
class SensitiveArea:
    id: str                    # Unique identifier (UUID4)
    timestamp: float           # Video timestamp in seconds
    bounding_box: BoundingBox  # Region coordinates
    area_type: str            # Type: "chatgpt", "atuin", etc.
    confidence: float = 1.0    # Confidence [0.0, 1.0]
    image_path: Optional[Path] = None  # Source frame image
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
```

**Validation Rules**:
- `confidence` must be between 0.0 and 1.0
- `area_type` must be in predefined SUPPORTED_AREA_TYPES
- `timestamp` must be non-negative
- `id` must be valid UUID format

### 3. DetectionResult

**Purpose**: Container for detection algorithm results.

**Design Decisions**:
- Support multiple detections per frame
- Include performance metrics (detection time)
- Track which detector produced the result
- Provide easy access to best detection

```python
@dataclass
class DetectionResult:
    regions: List[BoundingBox]     # Detected regions
    confidences: List[float]       # Per-region confidence scores
    detection_time: float          # Processing time in seconds
    detector_type: str             # Which detector produced this
    frame_metadata: Dict[str, Any] # Additional frame information
    timestamp: float               # Frame timestamp
```

**Key Methods**:
- `best_detection() -> Optional[Tuple[BoundingBox, float]]`: Highest confidence
- `filter_by_confidence(threshold: float) -> DetectionResult`: Filter results
- `merge_with(other: DetectionResult) -> DetectionResult`: Combine results
- `to_sensitive_areas(area_type: str) -> List[SensitiveArea]`: Convert format

### 4. BlurRegion

**Purpose**: Defines a temporal region to be blurred in the final video.

**Design Decisions**:
- Support time-based regions for temporal accuracy
- Configurable blur parameters per region
- Motion interpolation support for moving windows
- Confidence tracking for review workflow

```python
@dataclass
class BlurRegion:
    id: str                       # Unique identifier
    start_time: float             # Start timestamp (seconds)
    end_time: float              # End timestamp (seconds)
    bounding_box: BoundingBox    # Region to blur
    blur_type: str              # Filter: "gaussian", "pixelate", "composite"
    blur_strength: float = 1.0   # Strength multiplier [0.0, 2.0]
    interpolation: str = "linear" # Motion: "linear", "smooth", "none"
    confidence: float = 1.0      # Detection confidence
    needs_review: bool = False   # Flag for manual review
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Key Methods**:
- `duration() -> float`: Calculate time duration
- `overlaps_time(timestamp: float) -> bool`: Check temporal overlap
- `get_region_at_time(timestamp: float) -> BoundingBox`: Interpolated position
- `split_at_time(timestamp: float) -> Tuple[BlurRegion, BlurRegion]`: Split region

### 5. Timeline

**Purpose**: Complete timeline of blur operations for a video.

**Design Decisions**:
- Store video metadata for validation
- Support versioning for iterative refinement
- Provide efficient temporal queries
- Calculate statistics for reporting

```python
@dataclass
class Timeline:
    id: str                      # Unique identifier
    video_path: Path            # Source video file
    video_duration: float       # Total video length (seconds)
    frame_rate: float          # Video frame rate
    blur_regions: List[BlurRegion]  # All blur operations
    version: str = "1.0"       # Version for compatibility
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Key Methods**:
- `get_active_regions(timestamp: float) -> List[BlurRegion]`: Active at time
- `get_regions_in_range(start: float, end: float) -> List[BlurRegion]`: Time range
- `total_blur_duration() -> float`: Sum of all blur durations
- `blur_coverage_percentage() -> float`: Percentage of video blurred
- `optimize() -> Timeline`: Merge overlapping regions
- `validate() -> List[str]`: Check for issues

## Configuration Models

### DetectionConfig

**Purpose**: Configuration for detection algorithms.

```python
@dataclass
class DetectionConfig:
    frame_step: int = 10                    # Process every Nth frame
    confidence_threshold: float = 0.7       # Minimum confidence to accept
    max_regions_per_frame: int = 5         # Limit detections per frame
    color_space: str = "HSV"               # Color space for analysis
    template_match_method: str = "TM_CCOEFF_NORMED"
    
    # Color masking parameters
    color_lower_bound: Tuple[int, int, int] = (0, 0, 0)
    color_upper_bound: Tuple[int, int, int] = (180, 255, 255)
    
    # Temporal smoothing
    temporal_window: int = 5               # Frames for smoothing
    motion_threshold: float = 50.0         # Pixels to trigger tracking
    
    # Performance
    batch_size: int = 32                   # Frames per batch
    use_gpu: bool = False                  # GPU acceleration
```

### ProcessingConfig

**Purpose**: Configuration for video processing operations.

```python
@dataclass
class ProcessingConfig:
    quality_mode: str = "balanced"         # Processing quality level
    output_codec: str = "libx264"         # FFmpeg codec
    crf_value: int = 23                   # Constant Rate Factor
    preset: str = "medium"                # FFmpeg preset
    
    # Blur parameters
    gaussian_radius: int = 10             # Gaussian kernel size
    pixelate_factor: int = 10            # Pixelation block size
    noise_intensity: float = 0.1         # Random noise strength
    
    # Performance settings
    max_memory_gb: float = 8.0           # Memory limit
    max_threads: int = 4                 # Processing threads
    chunk_duration: float = 300.0        # Process in chunks (seconds)
    
    # Quality profiles
    quality_profiles: Dict[str, Dict] = field(default_factory=lambda: {
        'fast': {'frame_step': 30, 'crf': 28},
        'balanced': {'frame_step': 10, 'crf': 23},
        'high': {'frame_step': 5, 'crf': 18},
        'maximum': {'frame_step': 1, 'crf': 15}
    })
```

## Exception Hierarchy

**Design Philosophy**: Specific exceptions for different error types to enable precise error handling.

```python
class MiVideoEditorError(Exception):
    """Base exception for all video editor errors"""
    pass

class ValidationError(MiVideoEditorError):
    """Data validation errors"""
    pass

class DetectionError(MiVideoEditorError):
    """Detection algorithm errors"""
    pass

class ProcessingError(MiVideoEditorError):
    """Video processing errors"""
    pass

class StorageError(MiVideoEditorError):
    """Storage and database errors"""
    pass

class ConfigurationError(MiVideoEditorError):
    """Configuration and setup errors"""
    pass
```

## Validation System

**Purpose**: Centralized validation logic for all data models.

**Key Validators**:

```python
def validate_bounding_box(bbox: BoundingBox) -> None:
    """Validate bounding box constraints"""
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValidationError(f"Invalid box dimensions: {bbox.width}x{bbox.height}")
    if bbox.x < 0 or bbox.y < 0:
        raise ValidationError(f"Invalid box position: ({bbox.x}, {bbox.y})")

def validate_confidence(confidence: float) -> None:
    """Validate confidence score range"""
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError(f"Confidence must be [0.0, 1.0], got {confidence}")

def validate_timestamp(timestamp: float, video_duration: Optional[float] = None) -> None:
    """Validate timestamp constraints"""
    if timestamp < 0:
        raise ValidationError(f"Timestamp must be non-negative, got {timestamp}")
    if video_duration and timestamp > video_duration:
        raise ValidationError(f"Timestamp {timestamp} exceeds video duration {video_duration}")

def validate_area_type(area_type: str) -> None:
    """Validate area type against supported types"""
    if area_type not in SUPPORTED_AREA_TYPES:
        raise ValidationError(f"Unsupported area type: {area_type}")
```

## Serialization Strategy

**JSON Serialization**: All models support JSON serialization for API and storage.

**Design Decisions**:
- Use custom encoders for complex types (Path, datetime)
- Preserve type information for deserialization
- Support partial serialization for large objects
- Handle circular references gracefully

```python
class CoreJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for core models"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, BoundingBox):
            return {
                '_type': 'BoundingBox',
                'x': obj.x, 'y': obj.y,
                'width': obj.width, 'height': obj.height
            }
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

def deserialize_core_object(data: dict) -> Any:
    """Deserialize core objects from JSON"""
    if '_type' in data:
        if data['_type'] == 'BoundingBox':
            return BoundingBox(
                x=data['x'], y=data['y'],
                width=data['width'], height=data['height']
            )
    return data
```

## Constants and Enums

**System-wide constants** defined in `constants.py`:

```python
# Supported area types
SUPPORTED_AREA_TYPES = {
    'chatgpt': 'ChatGPT conversation interface',
    'atuin': 'Atuin terminal history search',
    'terminal': 'Generic terminal content',
    'custom': 'User-defined sensitive area'
}

# Blur filter types
BLUR_FILTER_TYPES = {
    'gaussian': 'Gaussian blur filter',
    'pixelate': 'Pixelation/mosaic filter',
    'noise': 'Random noise overlay',
    'composite': 'Combined blur effects'
}

# Quality modes
QUALITY_MODES = ['fast', 'balanced', 'high', 'maximum']

# Video constraints
MAX_VIDEO_DURATION = 14400  # 4 hours in seconds
MAX_FRAME_RESOLUTION = (3840, 2160)  # 4K resolution
MIN_FRAME_RESOLUTION = (640, 480)    # Minimum resolution

# Detection constraints
MIN_DETECTION_AREA = 100      # Minimum area in pixels
MAX_DETECTIONS_PER_FRAME = 20 # Maximum regions per frame
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Processing constraints
MAX_BLUR_REGIONS_PER_VIDEO = 1000
MAX_TIMELINE_DURATION_HOURS = 4
CHUNK_OVERLAP_SECONDS = 5.0
```

## Testing Strategy

**Unit Tests**: Each model should have comprehensive unit tests covering:
- Validation logic
- Serialization/deserialization
- Utility methods
- Edge cases and error conditions

**Property-Based Testing**: Use Hypothesis for testing with random data:
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0), st.integers(min_value=0),
       st.integers(min_value=1), st.integers(min_value=1))
def test_bounding_box_properties(x, y, width, height):
    bbox = BoundingBox(x, y, width, height)
    assert bbox.area == width * height
    assert bbox.center == (x + width // 2, y + height // 2)
```

## Performance Considerations

1. **Memory Efficiency**: Use `__slots__` for frequently created objects
2. **Lazy Loading**: Defer expensive operations until needed
3. **Caching**: Cache computed properties like area, center
4. **Batch Operations**: Support batch validation and serialization

## Future Extensions

**Planned Enhancements**:
1. Support for non-rectangular regions (polygons)
2. 3D bounding boxes for depth information
3. Confidence intervals instead of point estimates
4. Hierarchical region relationships (parent/child)
5. Automatic region merging strategies
6. Integration with machine learning frameworks

## Dependencies

**Required Dependencies**:
```python
# Core dependencies
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import uuid

# Validation
from abc import ABC, abstractmethod

# Optional dependencies
import numpy as np  # For numerical operations (if available)
```

**Design for Minimal Dependencies**: Core module should have minimal external dependencies to ensure stability and easy installation.

This design provides a solid foundation for the entire video editor system while maintaining flexibility for future enhancements.