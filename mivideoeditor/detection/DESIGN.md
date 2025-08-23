# Detection Module Design

## Overview

The `detection` module implements algorithms for automatically identifying sensitive areas in video frames. It uses a layered approach combining fast color-based pre-filtering with accurate template matching, designed to handle the specific use cases of ChatGPT interfaces and Atuin terminal history.

## Design Principles

1. **Hierarchical Detection**: Fast filtering followed by accurate matching
2. **Confidence-Based Results**: All detections include confidence scores
3. **Extensible Architecture**: Easy to add new detection algorithms
4. **Template Learning**: Learn from user annotations automatically
5. **Motion Awareness**: Handle moving windows and UI animations
6. **Performance Optimization**: Configurable speed vs. accuracy tradeoffs

## Module Structure

```
detection/
├── __init__.py
├── DESIGN.md
├── base.py              # Abstract base detector interface
├── template.py          # Template matching detector
├── color_mask.py        # Color-based region detector  
├── motion_tracker.py    # Motion tracking for moving regions
├── ensemble.py          # Combine multiple detectors
├── training.py          # Train detectors from annotations
├── utils.py            # Detection utilities and helpers
└── models/             # Trained model storage directory
```

## Detection Strategy

### Multi-Stage Detection Pipeline

```
Input Frame → Color Pre-filtering → Template Matching → Post-processing → Results
     ↓              ↓                    ↓               ↓            ↓
  1920x1080    Candidate Regions    Pattern Matching   NMS + Filter  BoundingBoxes
```

### Stage 1: Color Pre-filtering

**Purpose**: Quickly eliminate areas that can't contain target interfaces.

**Algorithm**: HSV color space filtering for UI-specific color signatures:
- **ChatGPT**: Dark sidebar (#1a1a1a), white chat bubbles (#ffffff), green assistant responses
- **Atuin**: Dark terminal background, white search overlay box

**Benefits**:
- 10-100x speed improvement over full-frame template matching
- Reduces false positives in irrelevant screen areas
- Adaptive to lighting and display variations

### Stage 2: Template Matching

**Purpose**: Accurate identification within candidate regions.

**Algorithm**: Normalized cross-correlation with learned templates:
- Multiple templates per interface (different states, sizes)
- Multi-scale matching for window size variations
- Rotation-invariant matching for slight screen tilts

**Template Types**:
- **Structural Templates**: UI layout patterns (sidebars, buttons)
- **Text Templates**: Consistent text elements ("ChatGPT", search prompts)
- **Icon Templates**: Interface icons and symbols

### Stage 3: Post-processing

**Purpose**: Refine detections and eliminate noise.

**Operations**:
- Non-maximum suppression to merge overlapping detections
- Temporal consistency checking across frames
- Size and aspect ratio validation
- Confidence threshold filtering

## Detector Implementations

### BaseDetector (Abstract Interface)

**Purpose**: Define contract for all detection algorithms.

```python
class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.is_trained = False
        self.templates = {}
        self.color_profiles = {}
    
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
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
    
    def batch_detect(self, frames: List[np.ndarray], 
                    timestamps: List[float]) -> List[DetectionResult]:
        """Detect regions in multiple frames (optimized batch processing)"""
        pass
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detector performance statistics"""
        pass
```

**Key Design Features**:
- Stateful training support
- Batch processing optimization
- Performance monitoring
- Model persistence
- Configuration management

### TemplateDetector

**Purpose**: Primary detector using template matching with color pre-filtering.

**Architecture**:
```python
class TemplateDetector(BaseDetector):
    """Template matching with color masking and multi-scale detection"""
    
    def __init__(self, config: DetectionConfig, area_type: str):
        super().__init__(config)
        self.area_type = area_type
        self.templates = {}           # Scale -> template image
        self.color_masks = []         # HSV range filters
        self.edge_templates = {}      # Edge-based templates
        self.feature_descriptors = {} # SIFT/ORB descriptors
        
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Multi-stage detection implementation"""
        start_time = time.time()
        
        # Stage 1: Color pre-filtering
        candidate_regions = self._apply_color_filters(frame)
        
        # Stage 2: Template matching in candidates
        detections = []
        for region in candidate_regions:
            matches = self._match_templates_in_region(frame, region)
            detections.extend(matches)
        
        # Stage 3: Post-processing
        filtered_detections = self._post_process_detections(detections, frame.shape)
        
        detection_time = time.time() - start_time
        
        return DetectionResult(
            regions=[d.bbox for d in filtered_detections],
            confidences=[d.confidence for d in filtered_detections],
            detection_time=detection_time,
            detector_type=f"template_{self.area_type}",
            frame_metadata={'candidate_regions': len(candidate_regions)},
            timestamp=timestamp
        )
```

**Key Features**:
- **Multi-scale Templates**: Handle different window sizes
- **Color Profile Learning**: Adapt to user's display settings
- **Edge Enhancement**: Improve matching on low-contrast interfaces
- **Negative Template Filtering**: Learn what NOT to detect

### ColorMaskDetector

**Purpose**: Fast detector for interfaces with consistent color signatures.

**Use Cases**:
- Quick detection when template matching is too slow
- Bootstrap detection for template learning
- Fallback when templates fail (partial occlusion)

**Algorithm**:
```python
def detect_by_color_signature(self, frame: np.ndarray) -> List[Detection]:
    """Detect regions based on color signatures"""
    
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply multiple color masks for different UI elements
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color_range in self.color_profiles:
        mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and filter by size/shape
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        # Filter by area and aspect ratio
        area = cv2.contourArea(contour)
        if area < self.config.min_detection_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Check if aspect ratio matches expected interface proportions
        if not self._validate_aspect_ratio(aspect_ratio):
            continue
            
        # Calculate confidence based on color match quality
        confidence = self._calculate_color_confidence(hsv[y:y+h, x:x+w])
        
        detections.append(Detection(
            bbox=BoundingBox(x, y, w, h),
            confidence=confidence,
            detection_type="color_mask"
        ))
    
    return detections
```

### MotionTracker

**Purpose**: Track detected regions across frames to handle motion.

**Key Challenges**:
- Window dragging (smooth motion)
- macOS animation effects (scale, fade, genie)
- Partial occlusion during movement
- Multi-object tracking when multiple windows visible

**Tracking Algorithm**:
```python
class MotionTracker:
    """Track regions across frames using multiple tracking algorithms"""
    
    def __init__(self, config: DetectionConfig):
        self.active_tracks = {}  # track_id -> Track object
        self.next_track_id = 0
        self.trackers = {
            'optical_flow': self._create_optical_flow_tracker,
            'kalman': self._create_kalman_tracker,
            'correlation': self._create_correlation_tracker
        }
    
    def update(self, frame: np.ndarray, detections: List[Detection], 
               timestamp: float) -> List[TrackedDetection]:
        """Update tracking with new frame and detections"""
        
        # Step 1: Predict current positions of existing tracks
        predictions = self._predict_track_positions(timestamp)
        
        # Step 2: Associate detections with existing tracks
        associations = self._associate_detections_to_tracks(detections, predictions)
        
        # Step 3: Update associated tracks
        for detection, track_id in associations:
            self.active_tracks[track_id].update(frame, detection, timestamp)
        
        # Step 4: Create new tracks for unassociated detections
        for detection in self._get_unassociated_detections(detections, associations):
            self._create_new_track(detection, timestamp)
        
        # Step 5: Remove lost tracks
        self._remove_lost_tracks(timestamp)
        
        # Step 6: Return tracked detections with motion information
        return self._get_tracked_detections()
```

**Track State Management**:
```python
@dataclass
class Track:
    """Individual object track"""
    id: int
    detections: List[Detection]        # Detection history
    timestamps: List[float]           # Timestamp history  
    predicted_position: BoundingBox   # Next predicted position
    velocity: Tuple[float, float]     # Motion velocity (px/sec)
    confidence: float                 # Track confidence
    frames_since_update: int          # Frames without detection
    state: str                        # "active", "lost", "occluded"
    
    def predict_next_position(self, dt: float) -> BoundingBox:
        """Predict position at future time"""
        pass
        
    def update_with_detection(self, detection: Detection, timestamp: float):
        """Update track with new detection"""
        pass
```

### EnsembleDetector

**Purpose**: Combine multiple detectors for improved accuracy and robustness.

**Ensemble Strategies**:

1. **Voting Ensemble**: Majority vote with confidence weighting
2. **Stacking Ensemble**: Use one detector's output as input to another
3. **Cascade Ensemble**: Fast detector filters candidates for slow detector

```python
class EnsembleDetector(BaseDetector):
    """Combine multiple detectors using ensemble methods"""
    
    def __init__(self, detectors: List[BaseDetector], strategy: str = "voting"):
        self.detectors = detectors
        self.strategy = strategy
        self.weights = self._calculate_detector_weights()
    
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Run ensemble detection"""
        
        # Get results from all detectors
        individual_results = []
        for detector in self.detectors:
            result = detector.detect(frame, timestamp)
            individual_results.append(result)
        
        # Combine results based on strategy
        if self.strategy == "voting":
            return self._voting_ensemble(individual_results)
        elif self.strategy == "cascade":
            return self._cascade_ensemble(frame, timestamp)
        elif self.strategy == "stacking":
            return self._stacking_ensemble(individual_results, frame)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")
    
    def _voting_ensemble(self, results: List[DetectionResult]) -> DetectionResult:
        """Weighted voting ensemble"""
        
        # Collect all detections with weights
        weighted_detections = []
        for i, result in enumerate(results):
            weight = self.weights[i]
            for bbox, conf in zip(result.regions, result.confidences):
                weighted_detections.append((bbox, conf * weight, i))
        
        # Cluster nearby detections
        clusters = self._cluster_detections(weighted_detections)
        
        # Generate final detections from clusters
        final_regions = []
        final_confidences = []
        
        for cluster in clusters:
            # Weighted average of bounding boxes
            avg_bbox = self._average_bounding_boxes(
                [(det[0], det[1]) for det in cluster]
            )
            
            # Combined confidence score
            combined_conf = sum(det[1] for det in cluster) / len(self.detectors)
            
            final_regions.append(avg_bbox)
            final_confidences.append(min(combined_conf, 1.0))
        
        return DetectionResult(
            regions=final_regions,
            confidences=final_confidences,
            detection_time=sum(r.detection_time for r in results),
            detector_type="ensemble_voting",
            frame_metadata={'individual_results': len(results)},
            timestamp=results[0].timestamp if results else 0.0
        )
```

## Training System

### Automatic Template Generation

**Purpose**: Learn templates automatically from user annotations.

**Process**:
1. **Template Extraction**: Extract sub-images from annotated regions
2. **Template Clustering**: Group similar templates together
3. **Template Selection**: Choose representative templates for each cluster
4. **Template Optimization**: Enhance templates for better matching

```python
class TemplateTrainer:
    """Train template detectors from annotations"""
    
    def train_from_annotations(self, annotations: List[SensitiveArea], 
                              frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate optimized templates from annotations"""
        
        # Step 1: Extract template candidates
        template_candidates = []
        for annotation in annotations:
            frame = frames[annotation.frame_index]
            bbox = annotation.bounding_box
            template = frame[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
            template_candidates.append((template, annotation.area_type))
        
        # Step 2: Cluster similar templates
        clusters = self._cluster_templates(template_candidates)
        
        # Step 3: Generate representative templates
        final_templates = {}
        for area_type, template_group in clusters.items():
            # Use median template to reduce noise
            representative = self._compute_median_template(template_group)
            
            # Enhance template for better matching
            enhanced = self._enhance_template(representative)
            
            final_templates[area_type] = enhanced
        
        return final_templates
    
    def _cluster_templates(self, templates: List[Tuple[np.ndarray, str]]) -> Dict[str, List[np.ndarray]]:
        """Cluster templates by similarity within each area type"""
        
        clusters = defaultdict(list)
        
        for template, area_type in templates:
            clusters[area_type].append(template)
        
        # Within each area type, sub-cluster by visual similarity
        final_clusters = {}
        for area_type, template_list in clusters.items():
            if len(template_list) <= 1:
                final_clusters[area_type] = template_list
                continue
                
            # Compute pairwise similarities
            similarities = self._compute_template_similarities(template_list)
            
            # Hierarchical clustering
            subclusters = self._hierarchical_clustering(template_list, similarities)
            
            # Take largest cluster as representative
            largest_cluster = max(subclusters, key=len)
            final_clusters[area_type] = largest_cluster
        
        return final_clusters
```

### Color Profile Learning

**Purpose**: Learn color signatures specific to user's setup.

```python
class ColorProfileLearner:
    """Learn color profiles from annotated regions"""
    
    def learn_color_profiles(self, annotations: List[SensitiveArea], 
                           frames: List[np.ndarray]) -> Dict[str, ColorProfile]:
        """Extract color profiles for each area type"""
        
        color_profiles = {}
        
        for area_type in set(ann.area_type for ann in annotations):
            # Get all regions for this area type
            regions = [(ann, frames[ann.frame_index]) for ann in annotations 
                      if ann.area_type == area_type]
            
            # Extract color statistics
            color_stats = self._extract_color_statistics(regions)
            
            # Generate HSV ranges that capture 95% of the data
            hsv_ranges = self._compute_hsv_ranges(color_stats)
            
            color_profiles[area_type] = ColorProfile(
                area_type=area_type,
                hsv_ranges=hsv_ranges,
                dominant_colors=color_stats['dominant_colors'],
                color_histogram=color_stats['histogram']
            )
        
        return color_profiles
    
    def _extract_color_statistics(self, regions: List[Tuple[SensitiveArea, np.ndarray]]) -> Dict:
        """Extract comprehensive color statistics from regions"""
        
        all_pixels = []
        dominant_colors = []
        
        for annotation, frame in regions:
            bbox = annotation.bounding_box
            region = frame[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
            
            # Convert to HSV
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Collect all pixels
            pixels = hsv_region.reshape(-1, 3)
            all_pixels.extend(pixels)
            
            # Find dominant colors using k-means
            dominant = self._find_dominant_colors(region, k=3)
            dominant_colors.extend(dominant)
        
        # Compute statistics
        all_pixels = np.array(all_pixels)
        
        return {
            'mean_hsv': np.mean(all_pixels, axis=0),
            'std_hsv': np.std(all_pixels, axis=0),
            'percentiles': {
                p: np.percentile(all_pixels, p, axis=0) 
                for p in [5, 25, 50, 75, 95]
            },
            'dominant_colors': dominant_colors,
            'histogram': cv2.calcHist([all_pixels.astype(np.uint8)], [0, 1, 2], 
                                    None, [180, 256, 256], [0, 180, 0, 256, 0, 256])
        }
```

## Performance Optimization

### Batch Processing

**Purpose**: Optimize detection across multiple frames.

**Optimizations**:
- GPU batch processing for template matching
- Frame pyramids for multi-scale detection
- Parallel processing across CPU cores
- Memory-efficient frame streaming

```python
class BatchDetectionProcessor:
    """Optimized batch processing for multiple frames"""
    
    def __init__(self, detector: BaseDetector, config: DetectionConfig):
        self.detector = detector
        self.config = config
        self.gpu_available = self._check_gpu_availability()
        
    def process_video_segment(self, video_path: Path, 
                             start_time: float, end_time: float) -> List[DetectionResult]:
        """Process a segment of video efficiently"""
        
        # Extract frames at specified intervals
        frames, timestamps = self._extract_frames(
            video_path, start_time, end_time, self.config.frame_step
        )
        
        if self.gpu_available and len(frames) > 32:
            # Use GPU batch processing for large batches
            return self._gpu_batch_detect(frames, timestamps)
        else:
            # Use CPU processing with parallelization
            return self._cpu_batch_detect(frames, timestamps)
    
    def _gpu_batch_detect(self, frames: List[np.ndarray], 
                         timestamps: List[float]) -> List[DetectionResult]:
        """GPU-accelerated batch detection"""
        
        # Upload frames to GPU
        gpu_frames = self._upload_to_gpu(frames)
        
        # Batch template matching
        batch_results = self._gpu_template_match_batch(gpu_frames)
        
        # Download results and post-process
        cpu_results = self._download_from_gpu(batch_results)
        
        return self._post_process_batch_results(cpu_results, timestamps)
```

### Adaptive Frame Sampling

**Purpose**: Dynamically adjust frame sampling based on scene complexity.

**Strategy**:
- High sampling rate during scene changes
- Low sampling rate during static scenes
- Boost sampling when confidence drops
- Reduce sampling when detections are consistent

```python
class AdaptiveSampler:
    """Dynamically adjust frame sampling rate"""
    
    def __init__(self, base_frame_step: int = 10):
        self.base_frame_step = base_frame_step
        self.recent_confidences = deque(maxlen=10)
        self.recent_scene_changes = deque(maxlen=5)
        
    def get_next_sampling_rate(self, last_result: DetectionResult, 
                              frame_diff: float) -> int:
        """Determine optimal sampling rate for next segment"""
        
        # Factor 1: Detection confidence
        if last_result.confidences:
            avg_confidence = np.mean(last_result.confidences)
            self.recent_confidences.append(avg_confidence)
        else:
            self.recent_confidences.append(0.0)
        
        # Factor 2: Scene change detection
        self.recent_scene_changes.append(frame_diff)
        
        # Calculate adaptive rate
        confidence_factor = self._calculate_confidence_factor()
        scene_change_factor = self._calculate_scene_change_factor()
        
        # Combine factors
        rate_multiplier = confidence_factor * scene_change_factor
        
        # Clamp to reasonable bounds
        new_rate = int(self.base_frame_step * rate_multiplier)
        return max(1, min(new_rate, 30))  # Between 1 and 30
    
    def _calculate_confidence_factor(self) -> float:
        """Lower confidence = higher sampling rate"""
        if not self.recent_confidences:
            return 1.0
            
        avg_conf = np.mean(self.recent_confidences)
        # Scale: confidence 0.9+ -> factor 1.0, confidence <0.5 -> factor 0.3
        return max(0.3, 2.0 - 2.0 * avg_conf)
    
    def _calculate_scene_change_factor(self) -> float:
        """More scene changes = higher sampling rate"""
        if not self.recent_scene_changes:
            return 1.0
            
        avg_change = np.mean(self.recent_scene_changes)
        # Scale: low change -> factor 1.0, high change -> factor 0.5
        normalized_change = min(avg_change / 0.3, 1.0)  # Normalize to [0,1]
        return max(0.5, 1.0 - 0.5 * normalized_change)
```

## Error Handling and Robustness

### Graceful Degradation

**Strategy**: Multiple fallback levels when detection fails.

```python
class RobustDetector(BaseDetector):
    """Detector with multiple fallback strategies"""
    
    def __init__(self, primary_detector: BaseDetector, 
                 fallback_detectors: List[BaseDetector]):
        self.primary = primary_detector
        self.fallbacks = fallback_detectors
        self.failure_counts = defaultdict(int)
        
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect with fallback strategy"""
        
        # Try primary detector
        try:
            result = self.primary.detect(frame, timestamp)
            if self._is_result_valid(result):
                return result
            else:
                self.failure_counts['primary_invalid'] += 1
        except Exception as e:
            logger.warning(f"Primary detector failed: {e}")
            self.failure_counts['primary_exception'] += 1
        
        # Try fallback detectors
        for i, fallback in enumerate(self.fallbacks):
            try:
                result = fallback.detect(frame, timestamp)
                if self._is_result_valid(result):
                    logger.info(f"Using fallback detector {i}")
                    return result
            except Exception as e:
                logger.warning(f"Fallback detector {i} failed: {e}")
                self.failure_counts[f'fallback_{i}'] += 1
        
        # All detectors failed - return empty result
        logger.error("All detectors failed, returning empty result")
        return DetectionResult(
            regions=[],
            confidences=[],
            detection_time=0.0,
            detector_type="failed",
            frame_metadata={'failure_counts': dict(self.failure_counts)},
            timestamp=timestamp
        )
```

## Testing and Validation

### Accuracy Metrics

**Evaluation Framework**: Comprehensive testing against labeled datasets.

```python
class DetectionEvaluator:
    """Evaluate detector accuracy and performance"""
    
    def evaluate_detector(self, detector: BaseDetector, 
                         test_dataset: List[Tuple[np.ndarray, List[BoundingBox]]]) -> Dict[str, float]:
        """Comprehensive detector evaluation"""
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        detection_times = []
        
        for frame, ground_truth_boxes in test_dataset:
            # Run detection
            start_time = time.time()
            result = detector.detect(frame)
            detection_times.append(time.time() - start_time)
            
            # Evaluate results
            tp, fp, fn = self._evaluate_frame_detections(result.regions, ground_truth_boxes)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_detection_time = np.mean(detection_times)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_detection_time': avg_detection_time,
            'total_detections': len(test_dataset),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
```

This comprehensive design provides a robust foundation for the detection module, balancing accuracy, performance, and extensibility while handling the specific requirements of your ChatGPT and Atuin detection use cases.