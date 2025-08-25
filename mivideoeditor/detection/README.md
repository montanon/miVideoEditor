# Detection Module

High-performance video detection system for identifying sensitive regions in screen recordings using template matching, color filtering, and motion tracking.

## Architecture

### Core Components

- **BaseDetector**: Abstract interface defining detection contract
- **TemplateDetector**: Fast detection using color pre-filtering + template matching (heuristic)
- **CNNDetector**: Deep learning detection using Convolutional Neural Networks
- **YOLODetector**: Real-time object detection using YOLO architecture
- **MotionTracker**: Temporal consistency across frames using Kalman filtering
- **EnsembleDetector**: Combines multiple detectors (voting/cascade/stacking)
- **DetectorTrainer**: Training pipeline for optimizing detection models

## Detection Pipeline

```
Frame → Color Filter → Template Match → Motion Track → Post-Process → Results
         ↓                ↓                ↓              ↓
      Candidates      Detections      Tracked IDs    Final Regions
```

## Quick Start

```python
from mivideoeditor.detection import DetectionConfig, TemplateDetector
import numpy as np

# Configure detector
config = DetectionConfig(
    confidence_threshold=0.7,
    max_regions_per_frame=5,
    enable_motion_tracking=True
)

# Create detector for ChatGPT interface
detector = TemplateDetector(config, area_type='chatgpt')

# Detect sensitive regions
frame = np.array(...)  # Your video frame
result = detector.detect(frame, timestamp=1.0)

# Access results
for bbox, confidence in zip(result.regions, result.confidences):
    print(f"Found region at {bbox} with {confidence:.2%} confidence")
```

## Motion Tracking

Track detections across frames for temporal consistency:

```python
from mivideoeditor.detection import MotionTracker

tracker = MotionTracker(config=config)

# Update with new detections
detections = [(bbox1, 0.9), (bbox2, 0.8)]
tracked = tracker.update(detections, timestamp=2.0)

# Each detection now has tracking info
for detection in tracked:
    print(f"Track {detection.track_id}: {detection.frames_tracked} frames")
```

## Ensemble Detection

Combine multiple detectors for improved accuracy:

```python
from mivideoeditor.detection import EnsembleDetector, EnsembleConfig

# Configure ensemble strategy
ensemble_config = EnsembleConfig(
    strategy='voting',  # or 'cascade', 'stacking'
    min_consensus_ratio=0.5,
    confidence_weights={'TemplateDetector': 0.7}
)

# Create ensemble with multiple detectors
ensemble = EnsembleDetector(
    detectors=[detector1, detector2],
    config=config,
    ensemble_config=ensemble_config
)

result = ensemble.detect(frame)
```

## Training System

Optimize detectors with annotated data:

```python
from mivideoeditor.detection import DetectorTrainer, TrainingConfig

# Configure training
training_config = TrainingConfig(
    min_annotations_per_type=5,
    validation_split=0.2,
    enable_data_augmentation=True
)

# Train detector
trainer = DetectorTrainer(config=training_config)
result = trainer.train_detector(
    detector=detector,
    annotations=annotations,
    save_path=Path('models/detector.pkl')
)

print(f"Training accuracy: {result.validation_accuracy}")
```

## Supported Area Types

- `chatgpt`: ChatGPT interface detection
- `atuin`: Terminal history (Atuin) detection
- Custom types can be added via configuration

## Performance

- **Color Pre-filtering**: 10-100x speedup over full-frame analysis
- **Template Matching**: Multi-scale matching with configurable scales
- **Motion Tracking**: Kalman filtering for smooth tracking
- **Batch Processing**: Optimized for processing multiple frames

## Configuration

Key parameters in `DetectionConfig`:

- `confidence_threshold`: Min confidence for valid detection (0.0-1.0)
- `max_regions_per_frame`: Limit detections per frame
- `min_detection_area`: Filter small false positives
- `frame_step`: Process every Nth frame for speed
- `enable_motion_tracking`: Track detections across frames
- `template_scales`: Scales for multi-scale template matching

## Detection Strategies

### Voting Ensemble
All detectors vote, weighted by confidence. Good for balanced accuracy.

### Cascade Ensemble
Fast detector filters candidates for slower, accurate detectors. Optimizes speed.

### Stacking Ensemble
Uses outputs from one detector as inputs to another. Best accuracy, slower.

## Error Handling

All detectors handle errors gracefully:

```python
try:
    result = detector.detect(frame)
except DetectionError as e:
    print(f"Detection failed: {e.error_code}")
    # Fallback to empty result
    result = create_detection_result_empty()
```

## Deep Learning Detection

### CNN-based Detection

For more accurate detection using deep learning:

```python
from mivideoeditor.detection.cnn import CNNDetector

# Create CNN detector
cnn_detector = CNNDetector(
    config=config,
    model_path=Path('models/chatgpt_cnn.pth'),
    area_type='chatgpt',
    use_gpu=True  # Enable GPU acceleration
)

# Detect with CNN
result = cnn_detector.detect(frame)

# Note: Training must be done independently
# cnn_detector.train(annotations)  # Not supported - use external training scripts
# cnn_detector.save_model(Path('models/trained_cnn.pth'))  # Only for saving loaded models
```

### YOLO Detection

For real-time object detection:

```python
from mivideoeditor.detection.cnn import YOLODetector

# Create YOLO detector
yolo = YOLODetector(
    config=config,
    model_version='yolov5s',  # or yolov5m, yolov5l, yolov5x
    area_type='general'
)

result = yolo.detect(frame)
```

### Hybrid Approach

Combine heuristic and CNN detectors for best results:

```python
# Fast heuristic for initial filtering
template_detector = TemplateDetector(config, 'chatgpt')

# Accurate CNN for validation (requires pre-trained model)
cnn_detector = CNNDetector(
    config, 
    model_path=Path('models/trained_model.pth'),  # Load pre-trained model
    use_gpu=True
)

# Ensemble: cascade strategy (fast → accurate)
ensemble = EnsembleDetector(
    detectors=[template_detector, cnn_detector],
    config=config,
    ensemble_config=EnsembleConfig(strategy='cascade')
)
```

## Detection Methods Comparison

| Method | Speed | Accuracy | GPU Required | Training Data | Training Support |
|--------|-------|----------|--------------|---------------|------------------|
| Template | ⚡⚡⚡ Fast | ⭐⭐ Good | No | Minimal | ✅ Built-in |
| CNN | ⚡ Slow | ⭐⭐⭐⭐ Excellent | Recommended | Large dataset | ❌ External only |
| YOLO | ⚡⚡ Medium | ⭐⭐⭐ Very Good | Recommended | Pre-trained | ❌ External only |
| Ensemble | ⚡⚡ Medium | ⭐⭐⭐⭐ Best | Optional | Varies | ✅ Built-in |

## Extending

Create custom detectors by inheriting `BaseDetector`:

```python
class CustomDetector(BaseDetector):
    def detect(self, frame: np.ndarray, timestamp: float) -> DetectionResult:
        # Your detection logic
        return DetectionResult(...)
    
    def train(self, annotations: list[SensitiveArea]) -> None:
        # Your training logic
        self.is_trained = True
```

## Dependencies

### Core Dependencies
- OpenCV (cv2): Image processing
- NumPy: Numerical operations
- Pydantic: Data validation
- Python 3.10+: Type hints and modern features

### Optional Dependencies (Deep Learning)
- **PyTorch**: Required for CNNDetector and YOLODetector
- **torchvision**: Computer vision utilities
- **ultralytics**: YOLO models (auto-installed via torch.hub)

```bash
# Install deep learning support
uv add torch torchvision

# For CUDA support (GPU acceleration)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

The detection module gracefully handles missing dependencies - CNN detectors will use mock implementations if PyTorch is not available.

## Training Architecture

### Built-in Training Support
- **TemplateDetector**: Supports training via the `train()` method using annotated data
- **EnsembleDetector**: Inherits training capabilities from constituent detectors

### External Training Required
- **CNNDetector**: Training must be done independently using external scripts
- **YOLODetector**: Fine-tuning requires external YOLO training tools

### CNN Training Workflow
```python
# 1. Train your model externally (e.g., using PyTorch training scripts)
# 2. Save the trained model to a .pth file
# 3. Load it in the detector

cnn_detector = CNNDetector(
    config=config,
    model_path=Path('models/my_trained_model.pth'),
    use_gpu=True
)

# The detector is now ready for inference
result = cnn_detector.detect(frame)
```

### YOLO Fine-tuning Workflow
```python
# 1. Prepare your dataset in YOLO format
# 2. Use YOLOv5/YOLOv8 training scripts to fine-tune
# 3. Load the fine-tuned weights (typically via torch.hub or direct loading)

yolo_detector = YOLODetector(
    config=config,
    model_version='path/to/custom/weights.pt'  # Custom weights
)
```

This separation allows for:
- Specialized training pipelines optimized for each model type
- Independent model development and validation
- Cleaner inference-only detection code
- Better resource management during deployment