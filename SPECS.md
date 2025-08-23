# Technical Specifications - Video Privacy Editor

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Data Structures](#data-structures)
3. [API Specifications](#api-specifications)
4. [File Formats](#file-formats)
5. [Algorithm Specifications](#algorithm-specifications)
6. [Performance Requirements](#performance-requirements)
7. [Integration Specifications](#integration-specifications)
8. [Testing Specifications](#testing-specifications)
9. [Deployment Specifications](#deployment-specifications)

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | High Performance |
|-----------|---------|-------------|------------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| Memory | 8GB RAM | 16GB RAM | 32GB RAM |
| Storage | 100GB free | 500GB free | 1TB SSD |
| GPU | None | Any discrete GPU | NVIDIA RTX series |
| Network | None | 100 Mbps | 1 Gbps |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8+ | Core runtime |
| FFmpeg | 4.4+ | Video processing |
| OpenCV | 4.5+ | Computer vision |
| SQLite | 3.35+ | Metadata storage |
| Node.js | 16+ | Web UI (optional) |

### Operating System Support

- **Primary**: macOS 11+, Ubuntu 20.04+
- **Secondary**: Windows 10+, CentOS 8+
- **Tested**: macOS 13 (Ventura), Ubuntu 22.04 LTS

## Data Structures

### Core Data Types

```python
@dataclass
class BoundingBox:
    """Represents a rectangular region in pixel coordinates"""
    x: int          # Left coordinate
    y: int          # Top coordinate
    width: int      # Box width
    height: int     # Box height
    
    def __post_init__(self):
        assert self.width > 0 and self.height > 0
        assert self.x >= 0 and self.y >= 0
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box"""
        # Implementation details...

@dataclass
class SensitiveArea:
    """Represents an annotated sensitive region"""
    id: str                    # Unique identifier
    timestamp: float           # Video timestamp in seconds
    bounding_box: BoundingBox  # Region coordinates
    area_type: str            # Type: "chatgpt", "atuin", etc.
    confidence: float = 1.0    # Confidence score [0.0, 1.0]
    image_path: Optional[Path] = None  # Path to extracted frame
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert 0.0 <= self.confidence <= 1.0
        assert self.area_type in SUPPORTED_AREA_TYPES

@dataclass
class DetectionResult:
    """Result from a detection algorithm"""
    regions: List[BoundingBox]     # Detected regions
    confidences: List[float]       # Per-region confidence
    detection_time: float          # Processing time in seconds
    detector_type: str             # Which detector produced this
    frame_metadata: Dict[str, Any] # Additional frame info
    
    @property
    def best_detection(self) -> Optional[Tuple[BoundingBox, float]]:
        """Returns highest confidence detection"""
        if not self.regions:
            return None
        best_idx = np.argmax(self.confidences)
        return self.regions[best_idx], self.confidences[best_idx]

@dataclass
class BlurRegion:
    """Represents a region to be blurred in final video"""
    start_time: float              # Start timestamp
    end_time: float               # End timestamp
    bounding_box: BoundingBox     # Region to blur
    blur_type: str               # Filter type: "gaussian", "pixelate", etc.
    blur_strength: float = 1.0   # Strength multiplier [0.0, 2.0]
    interpolation: str = "linear" # How to handle motion: "linear", "smooth"
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def overlaps_time(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

@dataclass
class Timeline:
    """Complete timeline of blur operations for a video"""
    video_path: Path
    video_duration: float         # Total video length in seconds
    frame_rate: float            # Video frame rate
    blur_regions: List[BlurRegion]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def get_active_regions(self, timestamp: float) -> List[BlurRegion]:
        """Get all blur regions active at given timestamp"""
        return [r for r in self.blur_regions if r.overlaps_time(timestamp)]
    
    def total_blur_duration(self) -> float:
        """Total seconds of video that will be blurred"""
        return sum(r.duration() for r in self.blur_regions)
```

### Configuration Objects

```python
@dataclass
class DetectionConfig:
    """Configuration for detection algorithms"""
    frame_step: int = 10              # Process every Nth frame
    confidence_threshold: float = 0.7  # Minimum confidence to accept
    max_regions_per_frame: int = 5    # Limit detections per frame
    color_space: str = "HSV"          # Color space for matching
    template_match_method: str = "TM_CCOEFF_NORMED"
    
    # Color masking parameters
    color_lower_bound: Tuple[int, int, int] = (0, 0, 0)
    color_upper_bound: Tuple[int, int, int] = (180, 255, 255)
    
    # Temporal smoothing
    temporal_window: int = 5          # Frames to consider for smoothing
    motion_threshold: float = 50.0    # Pixels moved to trigger tracking

@dataclass
class ProcessingConfig:
    """Configuration for video processing"""
    quality_mode: str = "balanced"    # "fast", "balanced", "high", "maximum"
    output_codec: str = "libx264"     # FFmpeg codec
    crf_value: int = 18              # Constant Rate Factor
    preset: str = "medium"           # FFmpeg preset
    
    # Blur parameters
    gaussian_radius: int = 10         # Gaussian blur kernel size
    pixelate_factor: int = 10        # Pixelation block size
    noise_intensity: float = 0.1     # Random noise strength
    
    # Performance
    max_memory_gb: float = 8.0       # Maximum memory usage
    max_threads: int = 4             # Processing threads
    chunk_duration: float = 300.0    # Process video in 5-minute chunks
```

## API Specifications

### Core Module APIs

#### Detection API

```python
class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect sensitive regions in a single frame"""
        pass
    
    @abstractmethod
    def train(self, annotations: List[SensitiveArea]) -> None:
        """Train detector on annotated data"""
        pass
    
    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Save trained model to disk"""
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Load trained model from disk"""
        pass
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """Detect regions in multiple frames (can be overridden for efficiency)"""
        return [self.detect(frame) for frame in frames]
    
    def validate_config(self, config: DetectionConfig) -> bool:
        """Validate configuration parameters"""
        pass

class TemplateDetector(BaseDetector):
    """Template matching with color pre-filtering"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.templates = {}
        self.color_masks = {}
    
    def add_template(self, name: str, template: np.ndarray) -> None:
        """Add a template for matching"""
        
    def set_color_mask(self, name: str, lower: Tuple, upper: Tuple) -> None:
        """Set color mask for template"""
        
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Implementation of template matching detection"""
        # 1. Apply color mask
        # 2. Perform template matching
        # 3. Filter results by confidence
        # 4. Apply non-maximum suppression
        pass
```

#### Timeline API

```python
class TimelineManager:
    """Manages blur timelines for videos"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.db = self._init_database()
    
    def create_timeline(self, video_path: Path) -> Timeline:
        """Create new timeline for video"""
        
    def load_timeline(self, video_path: Path) -> Optional[Timeline]:
        """Load existing timeline"""
        
    def save_timeline(self, timeline: Timeline) -> bool:
        """Save timeline to storage"""
        
    def build_from_detections(self, 
                            detections: List[DetectionResult],
                            config: ProcessingConfig) -> Timeline:
        """Build timeline from detection results"""
        
    def optimize_timeline(self, timeline: Timeline) -> Timeline:
        """Optimize timeline (merge overlapping regions, smooth motion)"""
        
    def validate_timeline(self, timeline: Timeline) -> List[str]:
        """Validate timeline and return any issues"""
```

#### Processing API

```python
class VideoProcessor:
    """Main video processing coordinator"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ffmpeg = FFmpegWrapper(config)
    
    def process_video(self, 
                     input_path: Path, 
                     timeline: Timeline, 
                     output_path: Path) -> ProcessingResult:
        """Process video with blur timeline"""
        
    def estimate_processing_time(self, 
                               video_path: Path, 
                               timeline: Timeline) -> float:
        """Estimate processing time in seconds"""
        
    def get_video_info(self, video_path: Path) -> VideoMetadata:
        """Extract video metadata using FFprobe"""

@dataclass
class ProcessingResult:
    """Result of video processing operation"""
    success: bool
    output_path: Optional[Path]
    processing_time: float
    original_size: int        # Bytes
    processed_size: int       # Bytes
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### Web API Specifications

#### REST Endpoints

```yaml
# OpenAPI 3.0 specification

openapi: 3.0.0
info:
  title: Video Privacy Editor API
  version: 1.0.0
  description: API for video annotation and processing

paths:
  /api/v1/videos:
    post:
      summary: Upload video for processing
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                video:
                  type: string
                  format: binary
      responses:
        201:
          description: Video uploaded successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VideoInfo'
  
  /api/v1/videos/{video_id}/frames:
    get:
      summary: Extract frames from video
      parameters:
        - name: video_id
          in: path
          required: true
          schema:
            type: string
        - name: timestamps
          in: query
          schema:
            type: array
            items:
              type: number
      responses:
        200:
          description: Frames extracted
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/FrameInfo'
  
  /api/v1/annotations:
    post:
      summary: Create annotation
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SensitiveArea'
      responses:
        201:
          description: Annotation created
  
  /api/v1/detections/{video_id}:
    post:
      summary: Run detection on video
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DetectionConfig'
      responses:
        202:
          description: Detection started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobInfo'
  
  /api/v1/timelines:
    post:
      summary: Generate blur timeline
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TimelineRequest'
      responses:
        201:
          description: Timeline generated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Timeline'
  
  /api/v1/process:
    post:
      summary: Process video with timeline
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ProcessingRequest'
      responses:
        202:
          description: Processing started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobInfo'

components:
  schemas:
    VideoInfo:
      type: object
      properties:
        id:
          type: string
        filename:
          type: string
        duration:
          type: number
        frame_rate:
          type: number
        resolution:
          type: object
          properties:
            width:
              type: integer
            height:
              type: integer
    
    FrameInfo:
      type: object
      properties:
        timestamp:
          type: number
        image_url:
          type: string
        frame_number:
          type: integer
    
    JobInfo:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [pending, running, completed, failed]
        progress:
          type: number
          minimum: 0
          maximum: 100
```

## File Formats

### Scene Definition (JSON)

```json
{
  "version": "1.0",
  "video_path": "/path/to/video.mp4",
  "video_metadata": {
    "duration": 7200.0,
    "frame_rate": 30.0,
    "resolution": {"width": 1920, "height": 1080}
  },
  "sensitive_scenes": [
    {
      "id": "scene_001",
      "start": "00:05:30.500",
      "end": "00:08:45.250", 
      "type": "chatgpt",
      "description": "ChatGPT conversation about sensitive project",
      "priority": "high",
      "metadata": {
        "marked_by": "user",
        "marked_at": "2024-01-15T10:30:00Z"
      }
    }
  ]
}
```

### Annotation Data (JSON)

```json
{
  "version": "1.0",
  "video_id": "video_123",
  "annotation_id": "ann_456",
  "timestamp": 330.5,
  "frame_number": 9915,
  "image_path": "/annotations/frames/frame_009915.png",
  "bounding_boxes": [
    {
      "id": "bb_001",
      "x": 200,
      "y": 150,
      "width": 800,
      "height": 600,
      "label": "chatgpt_conversation",
      "confidence": 1.0,
      "annotated_by": "user",
      "annotated_at": "2024-01-15T10:35:00Z"
    }
  ],
  "metadata": {
    "frame_quality": "good",
    "lighting_condition": "normal",
    "notes": "Clear ChatGPT interface visible"
  }
}
```

### Blur Timeline (JSON)

```json
{
  "version": "1.0",
  "video_path": "/path/to/video.mp4",
  "generated_at": "2024-01-15T11:00:00Z",
  "generator": {
    "tool": "miVideoEditor",
    "version": "1.0.0",
    "detector_types": ["template", "color_mask"]
  },
  "video_metadata": {
    "duration": 7200.0,
    "frame_rate": 30.0,
    "resolution": {"width": 1920, "height": 1080}
  },
  "blur_regions": [
    {
      "id": "blur_001",
      "start_time": 330.0,
      "end_time": 525.0,
      "region": {
        "x": 200,
        "y": 150,
        "width": 800,
        "height": 600
      },
      "blur_config": {
        "type": "composite",
        "gaussian_strength": 10,
        "pixelate_factor": 8,
        "noise_intensity": 0.15
      },
      "motion_tracking": {
        "enabled": true,
        "interpolation": "smooth",
        "keyframes": [
          {"time": 330.0, "x": 200, "y": 150},
          {"time": 350.0, "x": 220, "y": 155},
          {"time": 400.0, "x": 180, "y": 140}
        ]
      },
      "confidence": 0.95,
      "needs_review": false,
      "metadata": {
        "detected_by": "template_detector",
        "area_type": "chatgpt"
      }
    }
  ],
  "low_confidence_flags": [
    {
      "timestamp": 400.5,
      "region": {"x": 180, "y": 140, "width": 800, "height": 600},
      "confidence": 0.65,
      "reason": "partial_occlusion",
      "needs_review": true
    }
  ],
  "statistics": {
    "total_blur_duration": 195.0,
    "blur_percentage": 2.7,
    "regions_count": 1,
    "avg_confidence": 0.95
  }
}
```

### FFmpeg Filter Complex

```bash
# Generated filter complex for multiple blur regions
-filter_complex "
[0:v]split=3[main][blur1][blur2];

[blur1]crop=800:600:200:150[crop1];
[crop1]boxblur=10:5[gauss1];
[gauss1]scale=80:60[scale1];
[scale1]scale=800:600:flags=neighbor[pixel1];
[pixel1]noise=c0s=100:c0f=t+u[noise1];

[blur2]crop=400:300:500:400[crop2];
[crop2]boxblur=15:7[final2];

[main][noise1]overlay=200:150:enable='between(t,330,525)'[tmp1];
[tmp1][final2]overlay=500:400:enable='between(t,600,800)'[out]
" -map "[out]" -map 0:a
```

## Algorithm Specifications

### Template Matching Algorithm

```python
def template_match_with_color_mask(frame: np.ndarray, 
                                 template: np.ndarray,
                                 color_mask: ColorRange) -> List[Match]:
    """
    Multi-stage template matching algorithm
    
    Stage 1: Color pre-filtering
    Stage 2: Template matching  
    Stage 3: Non-maximum suppression
    Stage 4: Confidence scoring
    """
    
    # Stage 1: Color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_mask.lower, color_mask.upper)
    
    # Morphological operations to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    matches = []
    for contour in contours:
        # Skip small regions
        if cv2.contourArea(contour) < MIN_AREA_THRESHOLD:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract region for template matching
        region = frame[y:y+h, x:x+w]
        
        # Stage 2: Template matching within region
        if region.shape[0] >= template.shape[0] and region.shape[1] >= template.shape[1]:
            result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > TEMPLATE_THRESHOLD:
                # Adjust coordinates to original frame
                match_x = x + max_loc[0]
                match_y = y + max_loc[1]
                
                matches.append(Match(
                    x=match_x,
                    y=match_y,
                    width=template.shape[1],
                    height=template.shape[0],
                    confidence=max_val,
                    color_score=calculate_color_score(region, template)
                ))
    
    # Stage 3: Non-maximum suppression
    matches = apply_nms(matches, iou_threshold=0.3)
    
    # Stage 4: Final confidence scoring
    for match in matches:
        match.confidence = calculate_composite_score(
            template_score=match.confidence,
            color_score=match.color_score,
            size_score=calculate_size_score(match),
            position_score=calculate_position_score(match, frame.shape)
        )
    
    return sorted(matches, key=lambda m: m.confidence, reverse=True)

def calculate_composite_score(template_score: float,
                            color_score: float, 
                            size_score: float,
                            position_score: float) -> float:
    """Weighted combination of different confidence metrics"""
    weights = {
        'template': 0.5,
        'color': 0.25,
        'size': 0.15,
        'position': 0.1
    }
    
    return (weights['template'] * template_score +
            weights['color'] * color_score +
            weights['size'] * size_score +
            weights['position'] * position_score)
```

### Motion Tracking Algorithm

```python
def track_region_motion(frames: List[np.ndarray], 
                       initial_region: BoundingBox,
                       frame_timestamps: List[float]) -> List[Tuple[float, BoundingBox]]:
    """
    Track region motion across frames using optical flow
    """
    
    tracker_type = "CSRT"  # More accurate for partial occlusion
    tracker = cv2.TrackerCSRT_create()
    
    # Initialize tracker with first frame
    first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    bbox_tuple = (initial_region.x, initial_region.y, 
                 initial_region.width, initial_region.height)
    tracker.init(first_frame, bbox_tuple)
    
    tracked_regions = [(frame_timestamps[0], initial_region)]
    
    for i, (frame, timestamp) in enumerate(zip(frames[1:], frame_timestamps[1:]), 1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update tracker
        success, bbox = tracker.update(gray_frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            tracked_region = BoundingBox(x, y, w, h)
            tracked_regions.append((timestamp, tracked_region))
        else:
            # Tracking failed, use prediction
            predicted = predict_region_position(
                tracked_regions[-3:],  # Use last 3 positions
                timestamp
            )
            tracked_regions.append((timestamp, predicted))
            
            # Try to reinitialize tracker
            reinit_success = reinitialize_tracker(tracker, gray_frame, predicted)
            if not reinit_success:
                logger.warning(f"Tracking lost at frame {i}, using prediction")
    
    return smooth_trajectory(tracked_regions)

def smooth_trajectory(regions: List[Tuple[float, BoundingBox]]) -> List[Tuple[float, BoundingBox]]:
    """Apply Kalman filtering to smooth trajectory"""
    
    # Kalman filter for 2D position + velocity
    kalman = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurement vars
    
    # State: [x, y, dx, dy]
    kalman.statePre = np.array([regions[0][1].center[0], 
                               regions[0][1].center[1], 0, 0], dtype=np.float32)
    
    # Transition matrix (constant velocity model)
    dt = 1.0 / 30.0  # Assume 30 FPS
    kalman.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Measurement matrix
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    
    # Process and measurement noise
    kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
    kalman.measurementNoiseCov = 1.0 * np.eye(2, dtype=np.float32)
    kalman.errorCovPost = 1.0 * np.eye(4, dtype=np.float32)
    
    smoothed_regions = []
    
    for timestamp, region in regions:
        # Predict
        prediction = kalman.predict()
        
        # Update with measurement
        measurement = np.array([region.center[0], region.center[1]], dtype=np.float32)
        kalman.correct(measurement)
        
        # Get smoothed position
        smoothed_center = (int(kalman.statePost[0]), int(kalman.statePost[1]))
        smoothed_region = BoundingBox(
            smoothed_center[0] - region.width // 2,
            smoothed_center[1] - region.height // 2,
            region.width,
            region.height
        )
        
        smoothed_regions.append((timestamp, smoothed_region))
    
    return smoothed_regions
```

### Timeline Optimization Algorithm

```python
def optimize_timeline(blur_regions: List[BlurRegion]) -> List[BlurRegion]:
    """
    Optimize timeline by merging overlapping regions and smoothing transitions
    """
    
    # Sort by start time
    sorted_regions = sorted(blur_regions, key=lambda r: r.start_time)
    
    # Phase 1: Merge overlapping temporal regions
    merged_regions = []
    current_group = [sorted_regions[0]]
    
    for region in sorted_regions[1:]:
        if region.start_time <= current_group[-1].end_time + MERGE_THRESHOLD:
            # Overlapping or close enough to merge
            current_group.append(region)
        else:
            # Process current group and start new one
            merged_regions.append(merge_region_group(current_group))
            current_group = [region]
    
    # Process last group
    if current_group:
        merged_regions.append(merge_region_group(current_group))
    
    # Phase 2: Spatial optimization within each temporal segment
    optimized_regions = []
    for region in merged_regions:
        optimized = optimize_spatial_region(region)
        optimized_regions.append(optimized)
    
    # Phase 3: Add transition smoothing
    final_regions = add_transition_smoothing(optimized_regions)
    
    return final_regions

def merge_region_group(regions: List[BlurRegion]) -> BlurRegion:
    """Merge a group of temporally overlapping regions"""
    
    # Find temporal bounds
    start_time = min(r.start_time for r in regions)
    end_time = max(r.end_time for r in regions)
    
    # Find spatial union of all bounding boxes
    min_x = min(r.bounding_box.x for r in regions)
    min_y = min(r.bounding_box.y for r in regions)
    max_x = max(r.bounding_box.x + r.bounding_box.width for r in regions)
    max_y = max(r.bounding_box.y + r.bounding_box.height for r in regions)
    
    merged_bbox = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Use strongest blur type
    blur_types = [r.blur_type for r in regions]
    if "composite" in blur_types:
        blur_type = "composite"
    elif "gaussian" in blur_types:
        blur_type = "gaussian"
    else:
        blur_type = blur_types[0]
    
    # Average confidence
    avg_confidence = sum(r.confidence for r in regions) / len(regions)
    
    return BlurRegion(
        start_time=start_time,
        end_time=end_time,
        bounding_box=merged_bbox,
        blur_type=blur_type,
        blur_strength=1.0,
        confidence=avg_confidence
    )
```

## Performance Requirements

### Processing Speed Requirements

| Video Duration | Quality Mode | Target Processing Time | Memory Usage |
|----------------|--------------|------------------------|--------------|
| 1 hour | fast | 6 minutes | < 4GB |
| 1 hour | balanced | 15 minutes | < 6GB |
| 1 hour | high | 30 minutes | < 8GB |
| 1 hour | maximum | 120 minutes | < 12GB |

### Detection Accuracy Requirements

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Precision | 0.85 | 0.90 | 0.95 |
| Recall | 0.80 | 0.88 | 0.93 |
| F1 Score | 0.82 | 0.89 | 0.94 |
| False Positive Rate | < 0.15 | < 0.10 | < 0.05 |

### System Performance Metrics

```python
PERFORMANCE_THRESHOLDS = {
    'frame_extraction_fps': 100,      # Frames extracted per second
    'detection_fps': 5,               # Frames analyzed per second
    'video_processing_ratio': 0.5,    # Processing time / video duration
    'memory_efficiency': 0.8,         # Useful memory / total memory
    'disk_space_efficiency': 0.9,     # Output size / input size (max)
    
    # Response time requirements (web API)
    'api_response_time_p50': 200,     # 50th percentile (ms)
    'api_response_time_p95': 1000,    # 95th percentile (ms)
    'api_response_time_p99': 5000,    # 99th percentile (ms)
    
    # Concurrency
    'max_concurrent_jobs': 4,         # Parallel processing jobs
    'max_api_connections': 100,       # Concurrent API connections
}
```

## Integration Specifications

### FFmpeg Integration

```python
class FFmpegWrapper:
    """Wrapper for FFmpeg operations with proper error handling"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()
    
    def build_blur_command(self, 
                          input_path: Path,
                          timeline: Timeline,
                          output_path: Path) -> List[str]:
        """Build complete FFmpeg command for blur processing"""
        
        cmd = [
            str(self.ffmpeg_path),
            '-i', str(input_path),
            '-filter_complex', self._build_filter_complex(timeline),
            '-c:a', 'copy',  # Copy audio stream
            '-c:v', self.config.output_codec,
            '-crf', str(self.config.crf_value),
            '-preset', self.config.preset,
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        return cmd
    
    def _build_filter_complex(self, timeline: Timeline) -> str:
        """Generate filter_complex string for multiple blur regions"""
        
        if not timeline.blur_regions:
            return "[0:v]copy[out]"
        
        filter_parts = []
        overlay_chain = "[0:v]"
        
        for i, region in enumerate(timeline.blur_regions):
            # Create blur filter for this region
            blur_filter = self._create_blur_filter(region, i)
            filter_parts.append(blur_filter)
            
            # Add to overlay chain
            time_filter = f"enable='between(t,{region.start_time},{region.end_time})'"
            overlay_cmd = f"overlay={region.bounding_box.x}:{region.bounding_box.y}:{time_filter}"
            
            if i == 0:
                overlay_chain += f"[blur{i}]{overlay_cmd}[tmp{i}]"
            elif i == len(timeline.blur_regions) - 1:
                overlay_chain += f";[tmp{i-1}][blur{i}]{overlay_cmd}[out]"
            else:
                overlay_chain += f";[tmp{i-1}][blur{i}]{overlay_cmd}[tmp{i}]"
        
        return ";".join(filter_parts) + ";" + overlay_chain
    
    def _create_blur_filter(self, region: BlurRegion, index: int) -> str:
        """Create blur filter for specific region"""
        
        x, y = region.bounding_box.x, region.bounding_box.y
        w, h = region.bounding_box.width, region.bounding_box.height
        
        base_filter = f"[0:v]crop={w}:{h}:{x}:{y}"
        
        if region.blur_type == "gaussian":
            strength = int(region.blur_strength * 10)
            return f"{base_filter},boxblur={strength}:{strength//2}[blur{index}]"
            
        elif region.blur_type == "pixelate":
            factor = int(region.blur_strength * 10)
            return f"{base_filter},scale=iw/{factor}:ih/{factor},scale={w}:{h}:flags=neighbor[blur{index}]"
            
        elif region.blur_type == "composite":
            strength = int(region.blur_strength * 10)
            factor = max(2, int(region.blur_strength * 8))
            noise = int(region.blur_strength * 100)
            
            return (f"{base_filter}"
                   f",boxblur={strength}:{strength//2}"
                   f",scale=iw/{factor}:ih/{factor}"
                   f",scale={w}:{h}:flags=neighbor"
                   f",noise=c0s={noise}:c0f=t+u[blur{index}]")
        
        else:
            # Default to gaussian
            return f"{base_filter},boxblur=10:5[blur{index}]"
```

### Database Integration

```sql
-- Complete database schema for SQLite

-- Videos table
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    duration REAL NOT NULL,
    frame_rate REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    codec TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scenes table (sensitive time ranges)
CREATE TABLE scenes (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    scene_type TEXT NOT NULL,
    description TEXT,
    priority TEXT DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_time_range CHECK (end_time > start_time),
    CONSTRAINT valid_scene_type CHECK (scene_type IN ('chatgpt', 'atuin', 'terminal', 'custom'))
);

-- Annotations table (manual bounding box annotations)
CREATE TABLE annotations (
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
    CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0),
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Detections table (automated detection results)
CREATE TABLE detections (
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
    detection_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_bbox CHECK (bbox_width > 0 AND bbox_height > 0),
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Timelines table (blur timeline definitions)
CREATE TABLE timelines (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    version INTEGER DEFAULT 1,
    timeline_data JSON NOT NULL,
    status TEXT DEFAULT 'draft',
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_status CHECK (status IN ('draft', 'approved', 'processed'))
);

-- Processing jobs table
CREATE TABLE processing_jobs (
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
);

-- Indexes for performance
CREATE INDEX idx_videos_filepath ON videos(filepath);
CREATE INDEX idx_scenes_video_time ON scenes(video_id, start_time, end_time);
CREATE INDEX idx_annotations_video_timestamp ON annotations(video_id, timestamp);
CREATE INDEX idx_detections_video_timestamp ON detections(video_id, timestamp);
CREATE INDEX idx_detections_needs_review ON detections(needs_review) WHERE needs_review = TRUE;
CREATE INDEX idx_timelines_video_status ON timelines(video_id, status);
CREATE INDEX idx_jobs_status ON processing_jobs(status);

-- Views for common queries
CREATE VIEW video_stats AS
SELECT 
    v.id,
    v.filename,
    v.duration,
    COUNT(DISTINCT s.id) as scene_count,
    COUNT(DISTINCT a.id) as annotation_count,
    COUNT(DISTINCT d.id) as detection_count,
    COUNT(DISTINCT t.id) as timeline_count
FROM videos v
LEFT JOIN scenes s ON v.id = s.video_id
LEFT JOIN annotations a ON v.id = a.video_id
LEFT JOIN detections d ON v.id = d.video_id
LEFT JOIN timelines t ON v.id = t.video_id
GROUP BY v.id;

CREATE VIEW pending_reviews AS
SELECT 
    d.id,
    d.video_id,
    v.filename,
    d.timestamp,
    d.detector_type,
    d.confidence,
    d.bbox_x,
    d.bbox_y,
    d.bbox_width,
    d.bbox_height
FROM detections d
JOIN videos v ON d.video_id = v.id
WHERE d.needs_review = TRUE
ORDER BY d.confidence ASC, d.timestamp ASC;
```

## Testing Specifications

### Unit Test Requirements

```python
# Test coverage requirements
COVERAGE_THRESHOLDS = {
    'overall': 85,      # Overall code coverage
    'core_modules': 95, # Critical modules
    'detectors': 90,    # Detection algorithms
    'api': 80,         # Web API endpoints
    'utils': 70,       # Utility functions
}

class TestDetectorAccuracy:
    """Test detection algorithm accuracy"""
    
    @pytest.fixture
    def test_dataset(self):
        """Load standardized test dataset"""
        return load_test_dataset("test_data/validation_set/")
    
    def test_chatgpt_detection_precision(self, test_dataset):
        """Test ChatGPT detector precision"""
        detector = TemplateDetector("chatgpt")
        detector.load_model("models/chatgpt_detector.pkl")
        
        true_positives = 0
        false_positives = 0
        
        for frame, ground_truth in test_dataset.chatgpt_frames:
            detections = detector.detect(frame)
            
            for detection in detections:
                if any(detection.iou(gt) > 0.5 for gt in ground_truth):
                    true_positives += 1
                else:
                    false_positives += 1
        
        precision = true_positives / (true_positives + false_positives)
        assert precision >= 0.90, f"Precision {precision} below threshold"
    
    def test_detection_consistency(self):
        """Test detection consistency across similar frames"""
        detector = TemplateDetector("chatgpt")
        
        # Load sequence of similar frames
        frames = load_frame_sequence("test_data/sequences/chatgpt_stable/")
        
        detections = [detector.detect(frame) for frame in frames]
        
        # Check that detections are consistent (similar positions/sizes)
        for i in range(1, len(detections)):
            if detections[i-1].regions and detections[i].regions:
                position_drift = calculate_position_drift(
                    detections[i-1].regions[0],
                    detections[i].regions[0]
                )
                assert position_drift < 20, f"Excessive position drift: {position_drift}px"

class TestVideoProcessing:
    """Test video processing pipeline"""
    
    def test_processing_preserves_quality(self):
        """Test that processing doesn't degrade video quality excessively"""
        input_video = "test_data/sample_video.mp4"
        timeline = create_test_timeline()
        output_video = "test_output/processed.mp4"
        
        processor = VideoProcessor()
        result = processor.process_video(input_video, timeline, output_video)
        
        # Check quality metrics
        original_quality = calculate_video_quality(input_video)
        processed_quality = calculate_video_quality(output_video)
        
        quality_ratio = processed_quality / original_quality
        assert quality_ratio > 0.85, f"Quality degradation too high: {quality_ratio}"
    
    def test_blur_effectiveness(self):
        """Test that blur actually obscures sensitive information"""
        # Create test frame with known text
        test_frame = create_frame_with_text("SENSITIVE_INFO_12345")
        region = BoundingBox(100, 100, 200, 50)
        
        # Apply blur
        blur_filter = CompositeBlur()
        blurred_frame = blur_filter.apply(test_frame, region)
        
        # Test that text is not readable
        # Use OCR to verify text is obscured
        extracted_text = extract_text_ocr(blurred_frame)
        assert "SENSITIVE_INFO" not in extracted_text, "Blur ineffective - text still readable"
        
        # Test that adjacent areas are preserved
        adjacent_region = BoundingBox(350, 100, 200, 50)
        adjacent_text = extract_text_ocr(blurred_frame, adjacent_region)
        # This should still be readable if there was text there

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_large_video_processing(self):
        """Test processing of large video files"""
        # Create or use large test video (2+ hours)
        large_video = "test_data/large_sample.mp4"
        
        start_time = time.time()
        processor = VideoProcessor()
        
        # Monitor memory usage during processing
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        try:
            result = processor.process_video(
                large_video, 
                create_complex_timeline(),
                "test_output/large_processed.mp4"
            )
            
            processing_time = time.time() - start_time
            max_memory = memory_monitor.get_peak_usage()
            
            # Check performance requirements
            video_duration = get_video_duration(large_video)
            processing_ratio = processing_time / video_duration
            
            assert processing_ratio < 2.0, f"Processing too slow: {processing_ratio}x realtime"
            assert max_memory < 8 * 1024**3, f"Memory usage too high: {max_memory} bytes"
            
        finally:
            memory_monitor.stop()
    
    @pytest.mark.stress
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple videos"""
        import concurrent.futures
        
        videos = [f"test_data/video_{i}.mp4" for i in range(4)]
        
        def process_video_worker(video_path):
            processor = VideoProcessor()
            return processor.process_video(video_path, create_test_timeline(), 
                                         f"test_output/{Path(video_path).stem}_processed.mp4")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_video_worker, video) for video in videos]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Check that all processing completed successfully
        assert all(result.success for result in results), "Some processing jobs failed"
        
        # Check that concurrent processing was actually faster than sequential
        estimated_sequential_time = sum(result.processing_time for result in results)
        efficiency = total_time / estimated_sequential_time
        assert efficiency < 0.8, f"Concurrent processing not efficient: {efficiency}"
```

### Integration Test Requirements

```python
class TestAPIIntegration:
    """Test web API integration"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        from main import app
        return TestClient(app)
    
    def test_video_upload_and_processing_flow(self, client):
        """Test complete workflow through API"""
        
        # Step 1: Upload video
        with open("test_data/sample_video.mp4", "rb") as f:
            response = client.post("/api/v1/videos", 
                                 files={"video": ("sample.mp4", f, "video/mp4")})
        
        assert response.status_code == 201
        video_id = response.json()["id"]
        
        # Step 2: Create scene definition
        scenes_data = {
            "sensitive_scenes": [
                {"start": "00:00:30", "end": "00:01:00", "type": "chatgpt"}
            ]
        }
        response = client.post(f"/api/v1/videos/{video_id}/scenes", json=scenes_data)
        assert response.status_code == 201
        
        # Step 3: Extract frames for annotation
        response = client.get(f"/api/v1/videos/{video_id}/frames", 
                            params={"timestamps": [30.0, 45.0, 60.0]})
        assert response.status_code == 200
        frames = response.json()
        
        # Step 4: Create annotations
        for frame in frames:
            annotation_data = {
                "timestamp": frame["timestamp"],
                "bounding_box": {"x": 200, "y": 150, "width": 800, "height": 600},
                "label": "chatgpt"
            }
            response = client.post("/api/v1/annotations", json=annotation_data)
            assert response.status_code == 201
        
        # Step 5: Run detection
        detection_config = {
            "detector_type": "template",
            "confidence_threshold": 0.8
        }
        response = client.post(f"/api/v1/detections/{video_id}", json=detection_config)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # Step 6: Wait for detection completion
        for _ in range(30):  # Wait up to 30 seconds
            response = client.get(f"/api/v1/jobs/{job_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(1)
        else:
            pytest.fail("Detection job did not complete in time")
        
        # Step 7: Generate timeline
        response = client.post("/api/v1/timelines", json={"video_id": video_id})
        assert response.status_code == 201
        timeline_id = response.json()["id"]
        
        # Step 8: Process video
        processing_request = {
            "video_id": video_id,
            "timeline_id": timeline_id,
            "output_path": "test_output/api_processed.mp4"
        }
        response = client.post("/api/v1/process", json=processing_request)
        assert response.status_code == 202
        
        # Verify the complete pipeline worked
        assert Path("test_output/api_processed.mp4").exists()

class TestDatabaseIntegration:
    """Test database operations"""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session"""
        from storage import create_test_database
        return create_test_database()
    
    def test_annotation_crud_operations(self, db_session):
        """Test annotation CRUD operations"""
        storage = AnnotationStorage(db_session)
        
        # Create
        annotation = SensitiveArea(
            id="test_001",
            timestamp=30.0,
            bounding_box=BoundingBox(100, 100, 200, 150),
            area_type="chatgpt",
            confidence=0.95
        )
        
        saved_id = storage.save_annotation(annotation)
        assert saved_id == "test_001"
        
        # Read
        loaded_annotation = storage.load_annotation("test_001")
        assert loaded_annotation.timestamp == 30.0
        assert loaded_annotation.area_type == "chatgpt"
        
        # Update
        loaded_annotation.confidence = 0.98
        storage.save_annotation(loaded_annotation)
        
        updated_annotation = storage.load_annotation("test_001")
        assert updated_annotation.confidence == 0.98
        
        # Delete
        storage.delete_annotation("test_001")
        deleted_annotation = storage.load_annotation("test_001")
        assert deleted_annotation is None
```

## Deployment Specifications

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/videos /app/data/annotations /app/data/models /app/data/output

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV MAX_UPLOAD_SIZE=10737418240

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  video-editor:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - DATABASE_URL=sqlite:///app/data/app.db
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - video-editor
    restart: unless-stopped

volumes:
  data:
  config:
```

### Production Configuration

```yaml
# config/production.yaml
app:
  name: "Video Privacy Editor"
  version: "1.0.0"
  environment: "production"
  debug: false
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_connections: 1000
  keepalive_timeout: 65
  
database:
  url: "sqlite:///data/production.db"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  
storage:
  data_dir: "/app/data"
  max_file_size: "10GB"
  cleanup_interval: "24h"
  retention_days: 30
  
processing:
  max_concurrent_jobs: 4
  job_timeout: 7200  # 2 hours
  chunk_size: 300    # 5 minutes
  quality_profiles:
    fast:
      frame_step: 30
      crf: 28
    balanced:
      frame_step: 10
      crf: 23
    high:
      frame_step: 5
      crf: 18
    maximum:
      frame_step: 1
      crf: 15

logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/app.log"
  rotation: "1d"
  retention: "30d"
  
monitoring:
  metrics_enabled: true
  health_check_path: "/health"
  prometheus_path: "/metrics"
  
security:
  cors_origins: ["https://yourdomain.com"]
  max_upload_size: "10GB"
  rate_limiting:
    requests_per_minute: 100
    burst_size: 20
```

This completes the comprehensive technical specifications for the Video Privacy Editor system. The specifications cover all aspects from data structures to deployment, providing a complete blueprint for implementation.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create comprehensive README.md", "status": "completed", "activeForm": "Creating comprehensive README.md"}, {"content": "Create detailed DESIGN.md", "status": "completed", "activeForm": "Creating detailed DESIGN.md"}, {"content": "Create technical SPECS.md", "status": "completed", "activeForm": "Creating technical SPECS.md"}, {"content": "Build core data models", "status": "pending", "activeForm": "Building core data models"}, {"content": "Create detector interface and base implementation", "status": "pending", "activeForm": "Creating detector interface and base implementation"}, {"content": "Implement blur filter interface", "status": "pending", "activeForm": "Implementing blur filter interface"}, {"content": "Create timeline manager", "status": "pending", "activeForm": "Creating timeline manager"}, {"content": "Build annotation storage system", "status": "pending", "activeForm": "Building annotation storage system"}]