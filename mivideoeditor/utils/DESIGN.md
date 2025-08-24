# Utils Module Design

## Overview

The `utils` module provides common utilities, helper functions, and shared functionality used throughout the Video Privacy Editor system. It focuses on reusable components that don't fit into other specific modules but are essential for system operation.

## Design Principles

1. **Pure Functions**: Stateless, predictable functions where possible
2. **Single Responsibility**: Each utility has one clear purpose
3. **Performance Optimized**: Efficient implementations for commonly used operations
4. **Cross-Platform**: Compatible across macOS, Linux, and Windows
5. **Well Tested**: High test coverage for critical utilities
6. **Minimal Dependencies**: Reduce external dependency requirements
7. **Logging Integration**: Consistent logging throughout utilities

## Module Structure

```
utils/
├── __init__.py
├── DESIGN.md
├── video.py            # Video file operations and metadata
├── image.py            # Image processing utilities
├── time.py            # Time parsing and conversion utilities
├── validation.py      # Data validation helpers
├── logging_config.py  # Logging configuration and setup
├── config.py          # Configuration management utilities
├── file_utils.py      # File system operations
├── math_utils.py      # Mathematical computations
├── string_utils.py    # String processing utilities
├── system.py          # System information and platform detection
├── decorators.py      # Common decorators (caching, retry, etc.)
└── exceptions.py      # Utility-specific exceptions
```

## Core Components

### Video Utilities

**Purpose**: Handle video file operations, metadata extraction, and format validation.

```python
class VideoUtils:
    """Video file utilities and operations"""
    
    @staticmethod
    def get_video_info(video_path: Path) -> VideoInfo:
        """Extract comprehensive video metadata using FFprobe"""
        
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            audio_streams = []
            
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_streams.append(stream)
            
            if not video_stream:
                raise ValueError(f"No video stream found in {video_path}")
            
            # Parse video information
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Handle frame rate (can be fractional)
            fps_str = video_stream.get('avg_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Duration from format info
            duration = float(data['format'].get('duration', 0))
            
            return VideoInfo(
                filepath=video_path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                codec=video_stream.get('codec_name'),
                bitrate=int(video_stream.get('bit_rate', 0)),
                frame_count=int(video_stream.get('nb_frames', duration * fps)),
                has_audio=len(audio_streams) > 0,
                file_size=video_path.stat().st_size,
                container_format=data['format'].get('format_name'),
                metadata=data
            )
            
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(f"FFprobe failed: {e.stderr}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise VideoProcessingError(f"Failed to parse video info: {e}")
    
    @staticmethod
    def validate_video_file(video_path: Path) -> ValidationResult:
        """Validate video file format and accessibility"""
        
        errors = []
        warnings = []
        
        # Check file existence
        if not video_path.exists():
            errors.append(f"Video file not found: {video_path}")
            return ValidationResult(False, errors, warnings)
        
        # Check file size
        file_size = video_path.stat().st_size
        if file_size == 0:
            errors.append("Video file is empty")
        elif file_size > MAX_VIDEO_FILE_SIZE:
            warnings.append(f"Video file is large ({file_size / 1024**3:.1f} GB)")
        
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        if video_path.suffix.lower() not in valid_extensions:
            warnings.append(f"Uncommon video extension: {video_path.suffix}")
        
        # Try to read video metadata
        try:
            video_info = VideoUtils.get_video_info(video_path)
            
            # Validate video parameters
            if video_info.width < MIN_VIDEO_WIDTH or video_info.height < MIN_VIDEO_HEIGHT:
                errors.append(f"Video resolution too small: {video_info.width}x{video_info.height}")
            
            if video_info.duration > MAX_VIDEO_DURATION:
                warnings.append(f"Video is very long: {video_info.duration / 3600:.1f} hours")
            
            if video_info.fps > 120:
                warnings.append(f"Very high frame rate: {video_info.fps} fps")
            
        except Exception as e:
            errors.append(f"Cannot read video metadata: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def extract_frames_at_timestamps(video_path: Path, timestamps: List[float],
                                   output_dir: Path, quality: str = 'high') -> List[Path]:
        """Extract frames at specific timestamps"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_frames = []
        
        # Quality settings
        quality_settings = {
            'low': ['-q:v', '10'],      # Lower quality, smaller files
            'medium': ['-q:v', '5'],    # Balanced quality/size
            'high': ['-q:v', '2'],      # High quality
            'lossless': ['-q:v', '1']   # Best quality
        }
        
        for i, timestamp in enumerate(timestamps):
            frame_filename = f"frame_{i:06d}_{timestamp:.3f}s.png"
            frame_path = output_dir / frame_filename
            
            cmd = [
                'ffmpeg', '-ss', str(timestamp), '-i', str(video_path),
                '-frames:v', '1', '-f', 'image2'
            ] + quality_settings.get(quality, quality_settings['high']) + [
                '-y', str(frame_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, check=True)
                extracted_frames.append(frame_path)
                logger.debug(f"Extracted frame at {timestamp}s: {frame_path}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract frame at {timestamp}s: {e.stderr}")
                continue
        
        return extracted_frames
    
    @staticmethod
    def get_frame_at_timestamp(video_path: Path, timestamp: float) -> Optional[np.ndarray]:
        """Get single frame as numpy array"""
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            frames = VideoUtils.extract_frames_at_timestamps(
                video_path, [timestamp], temp_path.parent, quality='high'
            )
            
            if frames:
                frame_image = cv2.imread(str(frames[0]))
                return frame_image
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
            for frame_file in temp_path.parent.glob(f"frame_*_{timestamp:.3f}s.png"):
                frame_file.unlink()
        
        return None
    
    @staticmethod 
    def calculate_video_checksum(video_path: Path, algorithm: str = 'md5') -> str:
        """Calculate checksum for video file integrity verification"""
        
        import hashlib
        
        hash_obj = hashlib.new(algorithm)
        
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
```

### Image Utilities

**Purpose**: Image processing operations for frames and annotations.

```python
class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """Resize image with optional aspect ratio preservation"""
        
        if maintain_aspect:
            # Calculate scaling factor to fit within target size
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Add padding if needed to reach exact target size
            if (new_w, new_h) != target_size:
                # Create black image of target size
                padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                
                # Center the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return padded
            
            return resized
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def crop_image(image: np.ndarray, bbox: BoundingBox, 
                  padding: int = 0) -> np.ndarray:
        """Crop image to bounding box with optional padding"""
        
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(w, bbox.x + bbox.width + padding)
        y2 = min(h, bbox.y + bbox.height + padding)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray, 
                                 method: str = 'ssim') -> float:
        """Calculate similarity between two images"""
        
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        if method == 'ssim':
            from skimage.metrics import structural_similarity
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = img1, img2
            
            return structural_similarity(gray1, gray2)
        
        elif method == 'mse':
            # Mean Squared Error (lower is more similar)
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            # Convert to similarity score (0-1, higher is more similar)
            return 1.0 / (1.0 + mse / 10000)
        
        elif method == 'histogram':
            # Histogram correlation
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
        """Enhance image to improve detection accuracy"""
        
        enhanced = image.copy()
        
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen using unsharp mask
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    @staticmethod
    def detect_scene_change(img1: np.ndarray, img2: np.ndarray, 
                          threshold: float = 0.3) -> bool:
        """Detect if there's a significant scene change between frames"""
        
        # Calculate histogram difference
        hist_similarity = ImageUtils.calculate_image_similarity(
            img1, img2, method='histogram'
        )
        
        # Also check structural similarity
        ssim_similarity = ImageUtils.calculate_image_similarity(
            img1, img2, method='ssim'
        )
        
        # Combine metrics (scene change if either is below threshold)
        return hist_similarity < threshold or ssim_similarity < threshold
```

### Time Utilities

**Purpose**: Handle time parsing, conversion, and formatting operations.

```python
class TimeUtils:
    """Time parsing and conversion utilities"""
    
    @staticmethod
    def parse_time_string(time_str: str) -> float:
        """Parse various time string formats to seconds"""
        
        time_str = time_str.strip()
        
        # Format: HH:MM:SS.mmm or HH:MM:SS
        if ':' in time_str:
            parts = time_str.split(':')
            
            if len(parts) == 3:  # HH:MM:SS.mmm
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            
            elif len(parts) == 2:  # MM:SS.mmm
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        
        # Format: just seconds (123.456)
        try:
            return float(time_str)
        except ValueError:
            pass
        
        # Format: human readable (e.g., "1h 30m 45s")
        import re
        
        # Extract hours, minutes, seconds
        hours_match = re.search(r'(\d+(?:\.\d+)?)h', time_str, re.IGNORECASE)
        minutes_match = re.search(r'(\d+(?:\.\d+)?)m', time_str, re.IGNORECASE)
        seconds_match = re.search(r'(\d+(?:\.\d+)?)s', time_str, re.IGNORECASE)
        
        total_seconds = 0.0
        if hours_match:
            total_seconds += float(hours_match.group(1)) * 3600
        if minutes_match:
            total_seconds += float(minutes_match.group(1)) * 60
        if seconds_match:
            total_seconds += float(seconds_match.group(1))
        
        if total_seconds > 0:
            return total_seconds
        
        raise ValueError(f"Cannot parse time string: {time_str}")
    
    @staticmethod
    def format_duration(seconds: float, format_type: str = 'hms') -> str:
        """Format duration in seconds to human-readable string"""
        
        if format_type == 'hms':
            # Format: HH:MM:SS.mmm
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
            else:
                return f"{minutes:02d}:{secs:06.3f}"
        
        elif format_type == 'human':
            # Format: "1h 30m 45.5s"
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            
            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if secs > 0 or not parts:  # Always show seconds if no other parts
                parts.append(f"{secs:.1f}s")
            
            return " ".join(parts)
        
        elif format_type == 'compact':
            # Format: "1:30:45"
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes}:{secs:02d}"
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    @staticmethod
    def create_timestamp_range(start: float, end: float, 
                             step: float = 1.0) -> List[float]:
        """Create list of timestamps in a range"""
        
        timestamps = []
        current = start
        
        while current <= end:
            timestamps.append(round(current, 3))  # Round to milliseconds
            current += step
        
        # Ensure end timestamp is included
        if timestamps and timestamps[-1] != end:
            timestamps.append(end)
        
        return timestamps
    
    @staticmethod
    def find_time_overlap(range1: Tuple[float, float], 
                         range2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find overlap between two time ranges"""
        
        start1, end1 = range1
        start2, end2 = range2
        
        # Find overlap boundaries
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        # Check if there's actual overlap
        if overlap_start < overlap_end:
            return (overlap_start, overlap_end)
        
        return None
    
    @staticmethod
    def merge_overlapping_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping time ranges"""
        
        if not ranges:
            return []
        
        # Sort by start time
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]
        
        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # Check for overlap or adjacency
            if start <= last_end:
                # Merge ranges
                merged[-1] = (last_start, max(last_end, end))
            else:
                # No overlap, add as new range
                merged.append((start, end))
        
        return merged
```

### Validation Utilities

**Purpose**: Common validation functions for data integrity.

```python
class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_bounding_box(bbox: BoundingBox, 
                            frame_size: Optional[Tuple[int, int]] = None) -> ValidationResult:
        """Validate bounding box constraints"""
        
        errors = []
        warnings = []
        
        # Check dimensions
        if bbox.width <= 0 or bbox.height <= 0:
            errors.append(f"Invalid bounding box dimensions: {bbox.width}x{bbox.height}")
        
        # Check position
        if bbox.x < 0 or bbox.y < 0:
            errors.append(f"Negative bounding box position: ({bbox.x}, {bbox.y})")
        
        # Check against frame size if provided
        if frame_size:
            frame_w, frame_h = frame_size
            
            if bbox.x + bbox.width > frame_w:
                errors.append(f"Bounding box extends beyond frame width: {bbox.x + bbox.width} > {frame_w}")
            
            if bbox.y + bbox.height > frame_h:
                errors.append(f"Bounding box extends beyond frame height: {bbox.y + bbox.height} > {frame_h}")
            
            # Check if box is very small relative to frame
            box_area = bbox.area
            frame_area = frame_w * frame_h
            
            if box_area < frame_area * 0.001:  # Less than 0.1% of frame
                warnings.append(f"Bounding box is very small ({box_area} pixels)")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_confidence_score(confidence: float) -> ValidationResult:
        """Validate confidence score range"""
        
        errors = []
        warnings = []
        
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"Confidence score must be between 0.0 and 1.0, got {confidence}")
        
        if confidence < 0.5:
            warnings.append(f"Low confidence score: {confidence}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_timeline_consistency(timeline: Timeline) -> ValidationResult:
        """Validate timeline for consistency and logical issues"""
        
        errors = []
        warnings = []
        
        # Check timeline duration
        if timeline.video_duration <= 0:
            errors.append(f"Invalid video duration: {timeline.video_duration}")
        
        # Check frame rate
        if timeline.frame_rate <= 0 or timeline.frame_rate > 240:
            errors.append(f"Invalid frame rate: {timeline.frame_rate}")
        
        # Check blur regions
        for i, region in enumerate(timeline.blur_regions):
            # Time validation
            if region.start_time < 0:
                errors.append(f"Region {i}: Negative start time {region.start_time}")
            
            if region.end_time > timeline.video_duration:
                errors.append(f"Region {i}: End time {region.end_time} exceeds video duration")
            
            if region.start_time >= region.end_time:
                errors.append(f"Region {i}: Start time >= end time ({region.start_time} >= {region.end_time})")
            
            # Confidence validation
            conf_result = ValidationUtils.validate_confidence_score(region.confidence)
            if not conf_result.is_valid:
                errors.extend([f"Region {i}: {err}" for err in conf_result.errors])
            warnings.extend([f"Region {i}: {warn}" for warn in conf_result.warnings])
        
        # Check for excessive overlap
        overlap_count = 0
        for i, region1 in enumerate(timeline.blur_regions):
            for j, region2 in enumerate(timeline.blur_regions[i+1:], i+1):
                if TimeUtils.find_time_overlap(
                    (region1.start_time, region1.end_time),
                    (region2.start_time, region2.end_time)
                ):
                    overlap_count += 1
        
        if overlap_count > len(timeline.blur_regions) * 0.5:
            warnings.append(f"Many overlapping regions ({overlap_count} overlaps)")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        
        import re
        
        # Remove or replace invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            max_name_len = 255 - len(ext)
            sanitized = name[:max_name_len] + ext
        
        # Ensure not empty
        if not sanitized or sanitized.isspace():
            sanitized = "untitled"
        
        return sanitized
```

### System Utilities

**Purpose**: System information and platform-specific operations.

```python
class SystemUtils:
    """System information and platform utilities"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        import platform
        import psutil
        
        # Basic system info
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname': platform.node()
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        info.update({
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent
        })
        
        # CPU information
        info.update({
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else None
        })
        
        # Disk space for current directory
        disk_usage = psutil.disk_usage('.')
        info.update({
            'disk_total_gb': disk_usage.total / (1024**3),
            'disk_free_gb': disk_usage.free / (1024**3),
            'disk_percent_used': (disk_usage.total - disk_usage.free) / disk_usage.total * 100
        })
        
        return info
    
    @staticmethod
    def check_dependencies() -> Dict[str, Dict[str, Any]]:
        """Check availability and versions of required dependencies"""
        
        dependencies = {
            'ffmpeg': {'required': True, 'available': False, 'version': None},
            'ffprobe': {'required': True, 'available': False, 'version': None},
            'opencv': {'required': True, 'available': False, 'version': None},
            'numpy': {'required': True, 'available': False, 'version': None},
            'pillow': {'required': False, 'available': False, 'version': None},
            'scikit-image': {'required': False, 'available': False, 'version': None}
        }
        
        # Check FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies['ffmpeg']['available'] = True
                # Extract version from first line
                first_line = result.stdout.split('\n')[0]
                version_match = re.search(r'ffmpeg version (\S+)', first_line)
                if version_match:
                    dependencies['ffmpeg']['version'] = version_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check FFprobe
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies['ffprobe']['available'] = True
                first_line = result.stdout.split('\n')[0]
                version_match = re.search(r'ffprobe version (\S+)', first_line)
                if version_match:
                    dependencies['ffprobe']['version'] = version_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check Python packages
        python_packages = ['opencv', 'numpy', 'pillow', 'scikit-image']
        package_names = {
            'opencv': 'cv2',
            'pillow': 'PIL',
            'scikit-image': 'skimage'
        }
        
        for pkg in python_packages:
            try:
                import_name = package_names.get(pkg, pkg)
                module = __import__(import_name)
                dependencies[pkg]['available'] = True
                
                # Get version
                if hasattr(module, '__version__'):
                    dependencies[pkg]['version'] = module.__version__
                elif hasattr(module, 'version'):
                    dependencies[pkg]['version'] = str(module.version)
                    
            except ImportError:
                pass
        
        return dependencies
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information for hardware acceleration"""
        
        gpu_info = {
            'available': False,
            'devices': [],
            'cuda_available': False,
            'opencl_available': False
        }
        
        try:
            # Try to get NVIDIA GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                  '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        name, memory = line.split(', ')
                        gpu_info['devices'].append({
                            'name': name.strip(),
                            'memory_mb': int(memory),
                            'vendor': 'NVIDIA'
                        })
                        gpu_info['available'] = True
                        gpu_info['cuda_available'] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for other GPU vendors (simplified)
        if not gpu_info['devices']:
            # Try to detect AMD/Intel GPUs via system info
            system_info = SystemUtils.get_system_info()
            if 'AMD' in system_info.get('processor', '') or 'Radeon' in system_info.get('processor', ''):
                gpu_info['devices'].append({
                    'name': 'AMD GPU (detected)',
                    'vendor': 'AMD'
                })
                gpu_info['available'] = True
        
        return gpu_info
    
    @staticmethod
    def estimate_processing_performance(video_info: VideoInfo) -> Dict[str, float]:
        """Estimate processing performance for given video"""
        
        system_info = SystemUtils.get_system_info()
        gpu_info = SystemUtils.get_gpu_info()
        
        # Base performance factors
        cpu_factor = min(system_info['cpu_count_logical'], 8) / 4.0  # Normalize to 4 cores
        memory_factor = min(system_info['available_memory_gb'], 16) / 8.0  # Normalize to 8GB
        
        # GPU acceleration factor
        gpu_factor = 1.0
        if gpu_info['cuda_available']:
            gpu_factor = 2.0  # 2x speedup with CUDA
        elif gpu_info['available']:
            gpu_factor = 1.5  # 1.5x speedup with other GPU acceleration
        
        # Video complexity factor
        resolution_factor = (video_info.width * video_info.height) / (1920 * 1080)  # Normalize to 1080p
        fps_factor = video_info.fps / 30.0  # Normalize to 30fps
        
        # Estimate processing speed (multiple of realtime)
        base_speed = cpu_factor * memory_factor * gpu_factor
        video_adjusted_speed = base_speed / (resolution_factor * fps_factor)
        
        return {
            'cpu_factor': cpu_factor,
            'memory_factor': memory_factor,
            'gpu_factor': gpu_factor,
            'resolution_factor': resolution_factor,
            'fps_factor': fps_factor,
            'estimated_speed_multiple': max(0.1, video_adjusted_speed),  # At least 0.1x realtime
            'estimated_time_hours': video_info.duration / (3600 * max(0.1, video_adjusted_speed))
        }
```

### Decorators

**Purpose**: Common decorators for caching, retry logic, and performance monitoring.

```python
def retry(max_attempts: int = 3, delay: float = 1.0, 
         backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator

def cache_result(ttl: int = 3600, max_size: int = 128):
    """Cache function results with TTL"""
    
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            now = time.time()
            if key in cache and (now - cache_times[key]) < ttl:
                return cache[key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_times[key] = now
            
            # Cleanup old entries
            if len(cache) > max_size:
                oldest_key = min(cache_times.keys(), key=cache_times.get)
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        return wrapper
    return decorator

def measure_time(log_result: bool = True):
    """Measure and optionally log function execution time"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if log_result:
                    logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator

def validate_types(**type_checks):
    """Validate function argument types"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(f"{func.__name__}() argument '{param_name}' must be {expected_type.__name__}, got {type(value).__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
```

This comprehensive utils module design provides essential utilities and helper functions that support all other modules in the Video Privacy Editor system while maintaining high code quality, performance, and reliability.