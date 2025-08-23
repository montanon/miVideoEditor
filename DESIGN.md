# Design Document - Video Privacy Editor

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Detection Strategy](#detection-strategy)
5. [Processing Pipeline](#processing-pipeline)
6. [Storage Architecture](#storage-architecture)
7. [Web Interface Design](#web-interface-design)
8. [Performance Optimization](#performance-optimization)
9. [Extensibility](#extensibility)
10. [Security Considerations](#security-considerations)

## System Architecture

### Overview

The system is designed as a modular, multi-stage pipeline where each stage produces verifiable artifacts. This allows for human review and intervention at any point, ensuring accuracy while maintaining automation efficiency.

### Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Reviewability**: Every automated decision can be reviewed and overridden
3. **Extensibility**: New detectors and filters can be added without modifying core code
4. **Performance**: Configurable quality levels for different use cases
5. **Resilience**: Failures in one stage don't corrupt the entire pipeline

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                   (CLI / Web UI / API)                      │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                      │
│              (Pipeline Manager / Task Queue)                 │
└─────────────────────────────────────────────────────────────┘
                               │
┌──────────────┬──────────────┬──────────────┬───────────────┐
│  Annotation  │  Detection   │   Timeline   │   Processing  │
│   Module     │   Module     │   Module     │    Module     │
└──────────────┴──────────────┴──────────────┴───────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                        Storage Layer                         │
│           (File System / SQLite / Cache)                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Annotation Module

**Purpose**: Extract frames and collect training data through manual annotation.

**Components**:
- `FrameExtractor`: Extracts frames at specified timestamps
- `AnnotationUI`: Web-based interface for drawing bounding boxes
- `AnnotationStorage`: Persists annotations with metadata

**Key Design Decisions**:
- Use timestamp-based extraction for precise frame selection
- Store annotations separately from video for reusability
- Support multiple annotation formats (COCO, Pascal VOC, custom)

### 2. Detection Module

**Purpose**: Identify sensitive areas using trained models.

**Components**:
- `BaseDetector`: Abstract interface for all detectors
- `TemplateDetector`: Template matching with color masking
- `ColorMaskDetector`: HSV-based region detection
- `CNNDetector`: Deep learning-based detection (future)
- `DetectorEnsemble`: Combines multiple detectors

**Detection Strategy Hierarchy**:
```
1. Color Masking (fast, coarse filtering)
   ↓
2. Template Matching (accurate for known patterns)
   ↓
3. Feature Matching (handles variations)
   ↓
4. CNN Detection (most robust, future enhancement)
```

### 3. Timeline Module

**Purpose**: Manage temporal aspects of blur regions.

**Components**:
- `Timeline`: Core data structure for blur events
- `TimelineBuilder`: Constructs timelines from detections
- `TimelineOptimizer`: Merges and smooths regions
- `TimelineSerializer`: JSON export/import

**Key Features**:
- Temporal smoothing to prevent flicker
- Region interpolation for smooth transitions
- Confidence-based filtering
- Manual override support

### 4. Processing Module

**Purpose**: Apply blur effects to video using FFmpeg.

**Components**:
- `VideoProcessor`: Main processing coordinator
- `FFmpegWrapper`: FFmpeg command builder
- `BlurFilterChain`: Manages blur filter pipeline
- `QualityController`: Adjusts processing based on mode

**Processing Modes**:
```python
QUALITY_PROFILES = {
    'fast': {'frame_step': 30, 'blur_quality': 'low'},
    'balanced': {'frame_step': 10, 'blur_quality': 'medium'},
    'high': {'frame_step': 5, 'blur_quality': 'high'},
    'maximum': {'frame_step': 1, 'blur_quality': 'maximum'}
}
```

## Data Flow

### Stage 1: Scene Marking
```
User Input → scenes.json
{
  "sensitive_scenes": [
    {"start": "00:05:30", "end": "00:08:45", "type": "chatgpt"}
  ]
}
```

### Stage 2: Frame Extraction
```
Video + scenes.json → Extracted Frames
/annotations/
  ├── frame_00330.png  (5:30)
  ├── frame_00525.png  (8:45)
  └── metadata.json
```

### Stage 3: Annotation
```
Extracted Frames → Annotated Data
/annotations/
  ├── frame_00330.png
  ├── frame_00330.json  (bounding boxes)
  └── annotations.db    (SQLite)
```

### Stage 4: Detection
```
Video + Trained Model → Detection Results
{
  "detections": [
    {
      "timestamp": 330.0,
      "regions": [{"x": 200, "y": 150, "w": 800, "h": 600}],
      "confidence": 0.95
    }
  ]
}
```

### Stage 5: Timeline Generation
```
Detection Results → Blur Timeline
{
  "blur_regions": [
    {
      "start_time": 330.0,
      "end_time": 525.0,
      "regions": [...],
      "blur_filter": "composite"
    }
  ]
}
```

### Stage 6: Video Processing
```
Video + Timeline → Processed Video
FFmpeg filter_complex generation
Output with applied blur
```

## Detection Strategy

### Template Matching with Color Masking

**Rationale**: Combine speed of color filtering with accuracy of template matching.

**Algorithm**:
1. **Color Pre-filtering**:
   ```python
   # Extract color mask for ChatGPT interface
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   mask = cv2.inRange(hsv, lower_bound, upper_bound)
   ```

2. **Template Matching**:
   ```python
   # Match within masked regions only
   masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
   result = cv2.matchTemplate(masked_frame, template, cv2.TM_CCOEFF_NORMED)
   ```

3. **Confidence Scoring**:
   - Color match: 0.3 weight
   - Template match: 0.5 weight
   - Edge similarity: 0.2 weight

### Motion Tracking

**Approach**: Optical flow for smooth transitions

```python
# Track movement between frames
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, ...)
# Predict region movement
new_position = old_position + flow_vector
```

## Processing Pipeline

### FFmpeg Filter Construction

**Single Region**:
```bash
ffmpeg -i input.mp4 -filter_complex \
  "[0:v]crop=800:600:200:150,boxblur=10:5[blur]; \
   [0:v][blur]overlay=200:150" \
  output.mp4
```

**Multiple Regions with Composite Blur**:
```bash
# Gaussian + Pixelate + Noise
"[0:v]crop=w:h:x:y,boxblur=10:5,scale=iw/10:ih/10,scale=10*iw:10*ih:flags=neighbor,noise=c0s=100:c0f=t+u[blur]"
```

### Temporal Consistency

**Problem**: Frame-by-frame detection causes flicker

**Solution**: Temporal smoothing
```python
def smooth_timeline(detections, window=5):
    # Use sliding window to average positions
    # Interpolate missing detections
    # Apply Kalman filter for prediction
```

## Storage Architecture

### File Structure
```
project/
├── videos/
│   └── input.mp4
├── annotations/
│   ├── frames/
│   │   ├── frame_*.png
│   │   └── frame_*.json
│   └── annotations.db
├── models/
│   ├── chatgpt_detector.pkl
│   └── atuin_detector.pkl
├── timelines/
│   └── input_timeline.json
└── output/
    └── input_blurred.mp4
```

### Database Schema (SQLite)

```sql
-- Annotations table
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    video_path TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    label TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detections table
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    video_path TEXT NOT NULL,
    timestamp REAL NOT NULL,
    detector_type TEXT,
    regions JSON,
    confidence REAL,
    needs_review BOOLEAN DEFAULT FALSE
);

-- Timelines table
CREATE TABLE timelines (
    id INTEGER PRIMARY KEY,
    video_path TEXT NOT NULL,
    version INTEGER,
    timeline_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP
);
```

## Web Interface Design

### Annotation UI

**Technology Stack**:
- Frontend: React or Vue.js
- Backend: FastAPI
- Canvas: Fabric.js for drawing

**Features**:
- Video player with frame stepping
- Drawing tools (rectangle, polygon)
- Keyboard shortcuts for efficiency
- Batch annotation support

### Review UI

**Components**:
- Timeline visualization
- Confidence heat map
- Side-by-side comparison
- Edit tools for adjustments

**Mock UI Layout**:
```
┌─────────────────────────────────────────┐
│  Video Player  │  Detection Overlay      │
│                │  ┌──────────────┐       │
│                │  │ Confidence:   │       │
│                │  │ 0.95          │       │
│                │  └──────────────┘       │
├─────────────────────────────────────────┤
│  Timeline      [===|=========]           │
├─────────────────────────────────────────┤
│  Controls      │ Accept │ Reject │ Edit  │
└─────────────────────────────────────────┘
```

## Performance Optimization

### Frame Processing

**Parallel Processing**:
```python
from multiprocessing import Pool

def process_frame_batch(frames):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(detect_sensitive_areas, frames)
    return results
```

**GPU Acceleration** (when available):
```python
# Use OpenCV's CUDA modules
cv2.cuda.setDevice(0)
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(frame)
```

### Memory Management

**Strategies**:
1. Process video in chunks
2. Use memory-mapped files for large videos
3. Implement frame caching with LRU policy
4. Stream processing where possible

### FFmpeg Optimization

**Hardware Acceleration**:
```bash
# Use hardware encoding
ffmpeg -hwaccel videotoolbox -i input.mp4 ...  # macOS
ffmpeg -hwaccel cuda -i input.mp4 ...          # NVIDIA
```

**Multi-pass Processing**:
```bash
# First pass: Analysis
ffmpeg -i input.mp4 -an -pass 1 -f null /dev/null

# Second pass: Apply blur
ffmpeg -i input.mp4 -pass 2 output.mp4
```

## Extensibility

### Adding New Detectors

```python
class CustomDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.load_model()
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        # Implementation
        pass
    
    def train(self, annotations: List[Annotation]):
        # Training logic
        pass
```

### Adding New Blur Filters

```python
class CustomBlurFilter(BaseBlurFilter):
    def apply(self, image: np.ndarray, region: BoundingBox) -> np.ndarray:
        # Custom blur implementation
        return blurred_image
    
    def get_ffmpeg_filter(self, region: BoundingBox) -> str:
        # Return FFmpeg filter string
        return f"custom_filter=params"
```

### Plugin System

```python
# plugins.py
def load_plugins(plugin_dir):
    plugins = {}
    for file in Path(plugin_dir).glob("*.py"):
        module = import_module(file.stem)
        if hasattr(module, 'PLUGIN_CLASS'):
            plugins[module.PLUGIN_NAME] = module.PLUGIN_CLASS
    return plugins
```

## Security Considerations

### Input Validation

- Sanitize file paths to prevent directory traversal
- Validate video formats to prevent malicious files
- Limit file sizes to prevent DoS
- Validate JSON schemas for all configuration

### Privacy Protection

- Never log sensitive frame content
- Encrypt stored annotations if needed
- Provide secure deletion of processed data
- Implement access controls for web interface

### Process Isolation

```python
# Run FFmpeg in subprocess with limited permissions
subprocess.run(
    cmd,
    capture_output=True,
    timeout=max_timeout,
    env={'PATH': '/usr/bin:/bin'},  # Restricted PATH
    preexec_fn=lambda: os.setrlimit(...)  # Resource limits
)
```

## Error Handling

### Graceful Degradation

```python
class DetectionPipeline:
    def detect_with_fallback(self, frame):
        try:
            # Try primary detector
            return self.primary_detector.detect(frame)
        except DetectionError:
            # Fall back to simpler detector
            return self.fallback_detector.detect(frame)
        except Exception as e:
            # Log and return empty detection
            logger.error(f"Detection failed: {e}")
            return DetectionResult.empty()
```

### Recovery Strategies

1. **Checkpoint System**: Save progress periodically
2. **Partial Processing**: Continue with valid frames if some fail
3. **Retry Logic**: Automatic retry with exponential backoff
4. **Manual Override**: Allow user to skip problematic sections

## Future Enhancements

### Planned Features

1. **Real-time Processing**: Stream processing for live recordings
2. **Cloud Integration**: Process videos on cloud infrastructure
3. **ML Model Training**: Train custom models from annotations
4. **Advanced Motion Tracking**: Handle complex animations
5. **Multi-format Support**: Support for various video formats
6. **Collaborative Annotation**: Multiple users annotating simultaneously

### Architecture Preparations

- Use message queues for async processing
- Implement proper API versioning
- Design for horizontal scaling
- Prepare for containerization (Docker/Kubernetes)

## Testing Strategy

### Unit Tests
```python
def test_detector_accuracy():
    detector = TemplateDetector()
    frame = load_test_frame()
    result = detector.detect(frame)
    assert result.confidence > 0.9
```

### Integration Tests
```python
def test_pipeline_end_to_end():
    # Test complete flow from video to blurred output
    pipeline = ProcessingPipeline()
    output = pipeline.process(test_video)
    assert verify_blur_applied(output)
```

### Performance Tests
```python
def test_processing_speed():
    # Ensure processing meets performance targets
    start = time.time()
    process_video(large_video)
    duration = time.time() - start
    assert duration < expected_duration
```