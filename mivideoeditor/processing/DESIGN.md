# Processing Module Design

## Overview

The `processing` module handles all video processing operations, from applying blur filters to complete video rendering. It serves as the bridge between detection results and the final output video, with a focus on performance, quality, and destructive privacy protection.

## Design Principles

1. **Performance First**: Leverage FFmpeg and GPU acceleration when available
2. **Quality Preservation**: Maintain video quality while ensuring privacy
3. **Destructive Privacy**: Make sensitive information unrecoverable
4. **Modular Filters**: Composable blur effects that can be combined
5. **Progress Tracking**: Real-time progress reporting for long operations
6. **Error Recovery**: Graceful handling of processing failures
7. **Memory Efficiency**: Process large videos without excessive memory usage

## Module Structure

```
processing/
├── __init__.py
├── DESIGN.md
├── base.py              # Abstract interfaces for processing
├── video_processor.py   # Main video processing orchestrator
├── blur_filters/        # Blur filter implementations
│   ├── __init__.py
│   ├── base.py         # Abstract blur filter interface
│   ├── gaussian.py     # Gaussian blur filter
│   ├── pixelate.py     # Pixelation/mosaic filter
│   ├── noise.py        # Noise overlay filter
│   ├── composite.py    # Combined filters
│   └── motion_blur.py  # Motion-aware blur
├── ffmpeg_wrapper.py   # FFmpeg command generation and execution
├── quality_manager.py  # Quality profile management
├── progress_tracker.py # Progress monitoring and reporting
└── utils.py           # Processing utilities
```

## Processing Architecture

### Video Processing Pipeline

```
Timeline Input → Filter Chain Builder → FFmpeg Command Generator → Video Processor → Output Video
     ↓                    ↓                       ↓                    ↓              ↓
Region Analysis → Filter Optimization → Command Validation → Execution → Quality Check
```

### Multi-Stage Processing

```
Stage 1: Pre-processing
- Timeline validation
- Quality profile selection
- Memory allocation planning
- Temporary file creation

Stage 2: Filter Chain Construction
- Blur filter selection per region
- Filter parameter optimization
- Motion interpolation setup
- Temporal smoothing configuration

Stage 3: FFmpeg Command Generation
- Complex filter graph construction
- Hardware acceleration setup
- Output format configuration
- Error handling preparation

Stage 4: Video Processing
- Chunked processing for large files
- Progress monitoring
- Resource management
- Quality validation

Stage 5: Post-processing
- Output verification
- Cleanup operations
- Statistics collection
- Result reporting
```

## Core Components

### BaseVideoProcessor (Abstract Interface)

**Purpose**: Define the contract for all video processing operations.

```python
class BaseVideoProcessor(ABC):
    """Abstract base class for video processors"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.progress_tracker = ProgressTracker()
        self.quality_manager = QualityManager(config)
        self.temp_files = []
        
    @abstractmethod
    def process_video(self, input_path: Path, timeline: Timeline, 
                     output_path: Path) -> ProcessingResult:
        """Process video with blur timeline"""
        pass
        
    @abstractmethod
    def estimate_processing_time(self, video_path: Path, 
                               timeline: Timeline) -> float:
        """Estimate processing time in seconds"""
        pass
        
    @abstractmethod
    def get_supported_codecs(self) -> List[str]:
        """Get list of supported output codecs"""
        pass
        
    def validate_input(self, video_path: Path) -> ValidationResult:
        """Validate input video file"""
        pass
        
    def cleanup_temp_files(self) -> None:
        """Clean up temporary processing files"""
        pass
        
    def cancel_processing(self) -> bool:
        """Cancel ongoing processing operation"""
        pass
```

### FFmpegVideoProcessor

**Purpose**: Main video processor using FFmpeg for high-performance processing.

**Key Features**:
- **Hardware Acceleration**: Support for GPU encoding (NVENC, VideoToolbox, VAAPI)
- **Chunked Processing**: Handle large files by processing in segments
- **Complex Filter Graphs**: Generate sophisticated FFmpeg filter chains
- **Quality Optimization**: Automatic quality profile selection
- **Progress Monitoring**: Real-time progress reporting

```python
class FFmpegVideoProcessor(BaseVideoProcessor):
    """FFmpeg-based video processor with advanced features"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.ffmpeg_wrapper = FFmpegWrapper(config)
        self.filter_builder = FilterChainBuilder()
        self.chunk_manager = ChunkManager(config)
        
    def process_video(self, input_path: Path, timeline: Timeline, 
                     output_path: Path) -> ProcessingResult:
        """Process video with blur timeline"""
        
        start_time = time.time()
        
        try:
            # Stage 1: Validation and preparation
            self._validate_processing_inputs(input_path, timeline)
            
            # Stage 2: Quality profile selection
            quality_profile = self.quality_manager.select_profile(
                input_path, timeline, self.config.quality_mode
            )
            
            # Stage 3: Determine processing strategy
            video_info = self.ffmpeg_wrapper.get_video_info(input_path)
            processing_strategy = self._determine_processing_strategy(
                video_info, timeline, quality_profile
            )
            
            if processing_strategy == "single_pass":
                result = self._process_single_pass(input_path, timeline, 
                                                 output_path, quality_profile)
            elif processing_strategy == "chunked":
                result = self._process_chunked(input_path, timeline, 
                                             output_path, quality_profile)
            else:  # multi_pass
                result = self._process_multi_pass(input_path, timeline, 
                                                output_path, quality_profile)
            
            # Stage 4: Post-processing validation
            if result.success:
                validation_result = self._validate_output(output_path, timeline)
                if not validation_result.is_valid:
                    result.success = False
                    result.errors.extend(validation_result.errors)
            
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return ProcessingResult(
                success=False,
                output_path=None,
                processing_time=time.time() - start_time,
                original_size=input_path.stat().st_size if input_path.exists() else 0,
                processed_size=0,
                errors=[str(e)]
            )
        finally:
            self.cleanup_temp_files()
    
    def _process_single_pass(self, input_path: Path, timeline: Timeline, 
                           output_path: Path, quality_profile: Dict) -> ProcessingResult:
        """Process video in single pass for optimal quality"""
        
        # Build filter complex
        filter_complex = self.filter_builder.build_filter_chain(
            timeline, quality_profile
        )
        
        # Generate FFmpeg command
        cmd = self.ffmpeg_wrapper.build_processing_command(
            input_path=input_path,
            output_path=output_path,
            filter_complex=filter_complex,
            quality_profile=quality_profile
        )
        
        # Execute with progress monitoring
        result = self.ffmpeg_wrapper.execute_with_progress(
            cmd, 
            progress_callback=self.progress_tracker.update_progress
        )
        
        return ProcessingResult(
            success=result.returncode == 0,
            output_path=output_path if result.returncode == 0 else None,
            processing_time=result.execution_time,
            original_size=input_path.stat().st_size,
            processed_size=output_path.stat().st_size if output_path.exists() else 0,
            errors=result.stderr_lines if result.returncode != 0 else [],
            warnings=result.warning_lines
        )
    
    def _process_chunked(self, input_path: Path, timeline: Timeline, 
                        output_path: Path, quality_profile: Dict) -> ProcessingResult:
        """Process video in chunks to manage memory usage"""
        
        # Determine chunk boundaries
        chunks = self.chunk_manager.create_chunks(
            video_duration=timeline.video_duration,
            timeline=timeline,
            chunk_duration=self.config.chunk_duration
        )
        
        processed_chunks = []
        total_errors = []
        total_warnings = []
        
        try:
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
                
                # Create chunk timeline
                chunk_timeline = self._create_chunk_timeline(timeline, chunk)
                
                # Process chunk
                chunk_output = self._create_temp_file(f"chunk_{i:03d}.mp4")
                chunk_result = self._process_video_segment(
                    input_path, chunk_timeline, chunk_output, 
                    chunk.start_time, chunk.end_time, quality_profile
                )
                
                if not chunk_result.success:
                    raise ProcessingError(f"Chunk {i} processing failed: {chunk_result.errors}")
                
                processed_chunks.append(chunk_output)
                total_errors.extend(chunk_result.errors)
                total_warnings.extend(chunk_result.warnings)
                
                # Update progress
                progress = (i + 1) / len(chunks) * 100
                self.progress_tracker.update_progress(progress)
            
            # Concatenate chunks
            concat_result = self._concatenate_chunks(processed_chunks, output_path)
            
            if not concat_result.success:
                raise ProcessingError(f"Chunk concatenation failed: {concat_result.errors}")
            
            return ProcessingResult(
                success=True,
                output_path=output_path,
                processing_time=sum(chunk.processing_time for chunk in [chunk_result]),
                original_size=input_path.stat().st_size,
                processed_size=output_path.stat().st_size,
                errors=total_errors,
                warnings=total_warnings
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                original_size=input_path.stat().st_size,
                processed_size=0,
                errors=[str(e)] + total_errors
            )
        finally:
            # Cleanup chunk files
            for chunk_file in processed_chunks:
                if chunk_file.exists():
                    chunk_file.unlink()
```

## Blur Filter System

### BaseBlurFilter (Abstract Interface)

**Purpose**: Define interface for all blur filter implementations.

```python
class BaseBlurFilter(ABC):
    """Abstract base class for blur filters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filter_type = self.__class__.__name__.lower().replace('blur', '')
        
    @abstractmethod
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate FFmpeg filter string for this blur type"""
        pass
        
    @abstractmethod
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply blur to image region (for preview/testing)"""
        pass
        
    def validate_config(self) -> List[str]:
        """Validate filter configuration"""
        pass
        
    def get_filter_description(self) -> str:
        """Get human-readable description of filter"""
        pass
        
    def estimate_performance_impact(self, video_info: VideoInfo) -> float:
        """Estimate relative performance impact (1.0 = baseline)"""
        pass
```

### GaussianBlur

**Purpose**: Standard Gaussian blur for general-purpose privacy protection.

```python
class GaussianBlur(BaseBlurFilter):
    """Gaussian blur filter implementation"""
    
    def __init__(self, radius: int = 10, sigma: Optional[float] = None):
        config = {
            'radius': max(1, radius),
            'sigma': sigma or radius / 3.0
        }
        super().__init__(config)
    
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate FFmpeg boxblur filter"""
        
        radius = int(self.config['radius'] * region.blur_strength)
        
        # FFmpeg boxblur syntax: boxblur=luma_radius:luma_power:chroma_radius:chroma_power
        luma_radius = min(radius, 1023)  # FFmpeg limitation
        chroma_radius = luma_radius // 2
        
        return f"boxblur={luma_radius}:2:{chroma_radius}:2"
    
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply Gaussian blur to image region"""
        
        # Extract region
        y1, y2 = region.y, region.y + region.height
        x1, x2 = region.x, region.x + region.width
        
        # Validate bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image.shape[0], y2)
        x2 = min(image.shape[1], x2)
        
        # Apply blur
        sigma = self.config['sigma']
        kernel_size = int(2 * self.config['radius'] + 1)
        
        if kernel_size > 1:
            image[y1:y2, x1:x2] = cv2.GaussianBlur(
                image[y1:y2, x1:x2], 
                (kernel_size, kernel_size), 
                sigma
            )
        
        return image
    
    def estimate_performance_impact(self, video_info: VideoInfo) -> float:
        """Gaussian blur has moderate performance impact"""
        # Impact scales with radius squared
        radius_factor = (self.config['radius'] / 10) ** 2
        return 1.0 + (0.5 * radius_factor)
```

### PixelateBlur

**Purpose**: Pixelation/mosaic effect for strong privacy protection.

```python
class PixelateBlur(BaseBlurFilter):
    """Pixelation/mosaic blur filter"""
    
    def __init__(self, block_size: int = 16):
        config = {
            'block_size': max(2, block_size)
        }
        super().__init__(config)
    
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate FFmpeg pixelation filter using scale operations"""
        
        # Calculate pixelation factor based on block size and strength
        base_factor = self.config['block_size']
        strength_factor = max(1, int(base_factor * region.blur_strength))
        
        # Region dimensions
        w, h = region.bounding_box.width, region.bounding_box.height
        
        # Calculate downscaled dimensions
        down_w = max(1, w // strength_factor)
        down_h = max(1, h // strength_factor)
        
        # Create pixelation effect: scale down then scale back up with nearest neighbor
        return (f"scale={down_w}:{down_h}:flags=area,"
                f"scale={w}:{h}:flags=neighbor")
    
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply pixelation to image region"""
        
        # Extract region
        y1, y2 = region.y, region.y + region.height
        x1, x2 = region.x, region.x + region.width
        
        # Validate bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image.shape[0], y2)
        x2 = min(image.shape[1], x2)
        
        region_img = image[y1:y2, x1:x2].copy()
        
        if region_img.size > 0:
            # Calculate pixelation dimensions
            block_size = self.config['block_size']
            h, w = region_img.shape[:2]
            
            # Downscale
            small_h = max(1, h // block_size)
            small_w = max(1, w // block_size)
            
            small_img = cv2.resize(region_img, (small_w, small_h), 
                                 interpolation=cv2.INTER_AREA)
            
            # Upscale with nearest neighbor for pixelated effect
            pixelated = cv2.resize(small_img, (w, h), 
                                 interpolation=cv2.INTER_NEAREST)
            
            image[y1:y2, x1:x2] = pixelated
        
        return image
    
    def estimate_performance_impact(self, video_info: VideoInfo) -> float:
        """Pixelation has low performance impact"""
        return 0.8  # Faster than Gaussian blur
```

### NoiseOverlay

**Purpose**: Add random noise to make content unrecoverable.

```python
class NoiseOverlay(BaseBlurFilter):
    """Random noise overlay filter for maximum privacy"""
    
    def __init__(self, intensity: float = 0.3, noise_type: str = "gaussian"):
        config = {
            'intensity': max(0.0, min(1.0, intensity)),
            'noise_type': noise_type  # "gaussian", "uniform", "salt_pepper"
        }
        super().__init__(config)
    
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate FFmpeg noise filter"""
        
        # FFmpeg noise filter parameters
        intensity = int(self.config['intensity'] * region.blur_strength * 100)
        
        # Different noise patterns
        if self.config['noise_type'] == "gaussian":
            return f"noise=c0s={intensity}:c0f=t+u:c1s={intensity}:c1f=t+u:c2s={intensity}:c2f=t+u"
        elif self.config['noise_type'] == "uniform":
            return f"noise=c0s={intensity}:c0f=u:c1s={intensity}:c1f=u:c2s={intensity}:c2f=u"
        else:  # salt_pepper
            return f"noise=c0s={intensity}:c0f=p:c1s={intensity}:c1f=p:c2s={intensity}:c2f=p"
    
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply noise overlay to image region"""
        
        # Extract region
        y1, y2 = region.y, region.y + region.height
        x1, x2 = region.x, region.x + region.width
        
        # Validate bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image.shape[0], y2)
        x2 = min(image.shape[1], x2)
        
        region_img = image[y1:y2, x1:x2]
        
        if region_img.size > 0:
            intensity = self.config['intensity']
            
            if self.config['noise_type'] == "gaussian":
                # Gaussian noise
                noise = np.random.normal(0, intensity * 255, region_img.shape).astype(np.int16)
                noisy = np.clip(region_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
            elif self.config['noise_type'] == "uniform":
                # Uniform noise
                noise = np.random.uniform(-intensity * 255, intensity * 255, region_img.shape).astype(np.int16)
                noisy = np.clip(region_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
            else:  # salt_pepper
                # Salt and pepper noise
                noisy = region_img.copy()
                salt_pepper_ratio = intensity * 0.1  # 10% max coverage
                
                # Salt noise (white pixels)
                salt_mask = np.random.random(region_img.shape[:2]) < salt_pepper_ratio
                noisy[salt_mask] = 255
                
                # Pepper noise (black pixels)
                pepper_mask = np.random.random(region_img.shape[:2]) < salt_pepper_ratio
                noisy[pepper_mask] = 0
            
            image[y1:y2, x1:x2] = noisy
        
        return image
```

### CompositeBlur

**Purpose**: Combine multiple blur effects for maximum privacy protection.

```python
class CompositeBlur(BaseBlurFilter):
    """Composite blur combining multiple effects"""
    
    def __init__(self, effects: Optional[List[Dict]] = None):
        if effects is None:
            effects = [
                {'type': 'gaussian', 'radius': 12},
                {'type': 'pixelate', 'block_size': 8},
                {'type': 'noise', 'intensity': 0.15, 'noise_type': 'gaussian'}
            ]
        
        config = {'effects': effects}
        super().__init__(config)
        
        # Initialize component filters
        self.component_filters = []
        for effect in effects:
            if effect['type'] == 'gaussian':
                self.component_filters.append(
                    GaussianBlur(radius=effect.get('radius', 10))
                )
            elif effect['type'] == 'pixelate':
                self.component_filters.append(
                    PixelateBlur(block_size=effect.get('block_size', 16))
                )
            elif effect['type'] == 'noise':
                self.component_filters.append(
                    NoiseOverlay(
                        intensity=effect.get('intensity', 0.3),
                        noise_type=effect.get('noise_type', 'gaussian')
                    )
                )
    
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate composite FFmpeg filter chain"""
        
        filter_parts = []
        for component_filter in self.component_filters:
            filter_str = component_filter.get_ffmpeg_filter(region, video_info)
            filter_parts.append(filter_str)
        
        # Chain filters together
        return ",".join(filter_parts)
    
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply all component filters sequentially"""
        
        result_image = image.copy()
        
        # Apply each filter in sequence
        for component_filter in self.component_filters:
            result_image = component_filter.apply_to_image(result_image, region)
        
        return result_image
    
    def estimate_performance_impact(self, video_info: VideoInfo) -> float:
        """Sum of all component filter impacts"""
        total_impact = 0.0
        for component_filter in self.component_filters:
            total_impact += component_filter.estimate_performance_impact(video_info)
        
        return total_impact
```

### MotionBlur

**Purpose**: Apply motion-aware blur for moving regions.

```python
class MotionBlur(BaseBlurFilter):
    """Motion-aware blur filter"""
    
    def __init__(self, angle: float = 0.0, length: int = 15):
        config = {
            'angle': angle % 360,  # Motion direction in degrees
            'length': max(1, length)  # Blur length in pixels
        }
        super().__init__(config)
    
    def get_ffmpeg_filter(self, region: BlurRegion, 
                         video_info: VideoInfo) -> str:
        """Generate motion blur filter"""
        
        # Calculate motion blur kernel
        angle_rad = np.radians(self.config['angle'])
        length = int(self.config['length'] * region.blur_strength)
        
        # Create motion blur kernel coordinates
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        
        # Use a combination of directional blur filters
        if abs(dx) > abs(dy):  # Horizontal motion dominant
            return f"boxblur={abs(dx)}:1:0:0"
        else:  # Vertical motion dominant
            return f"boxblur=0:0:{abs(dy)}:1"
    
    def apply_to_image(self, image: np.ndarray, 
                      region: BoundingBox) -> np.ndarray:
        """Apply motion blur to image region"""
        
        # Extract region
        y1, y2 = region.y, region.y + region.height
        x1, x2 = region.x, region.x + region.width
        
        # Validate bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image.shape[0], y2)
        x2 = min(image.shape[1], x2)
        
        region_img = image[y1:y2, x1:x2]
        
        if region_img.size > 0:
            # Create motion blur kernel
            angle = self.config['angle']
            length = self.config['length']
            
            # Calculate kernel
            kernel = self._create_motion_kernel(angle, length)
            
            # Apply convolution
            blurred = cv2.filter2D(region_img, -1, kernel)
            image[y1:y2, x1:x2] = blurred
        
        return image
    
    def _create_motion_kernel(self, angle: float, length: int) -> np.ndarray:
        """Create motion blur kernel"""
        
        # Create kernel matrix
        kernel_size = length * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate line coordinates
        center = length
        angle_rad = np.radians(angle)
        
        for i in range(length + 1):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
            
            if i > 0:  # Mirror for negative direction
                x = int(center - i * np.cos(angle_rad))
                y = int(center - i * np.sin(angle_rad))
                
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0
        
        # Normalize kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        
        return kernel
```

## FFmpeg Integration

### FFmpegWrapper

**Purpose**: Generate and execute FFmpeg commands with proper error handling.

```python
class FFmpegWrapper:
    """Wrapper for FFmpeg operations"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ffmpeg_path = self._find_ffmpeg_executable()
        self.ffprobe_path = self._find_ffprobe_executable()
        self.hardware_acceleration = self._detect_hardware_acceleration()
        
    def build_processing_command(self, input_path: Path, output_path: Path,
                               filter_complex: str, quality_profile: Dict) -> List[str]:
        """Build complete FFmpeg command for video processing"""
        
        cmd = [str(self.ffmpeg_path)]
        
        # Hardware acceleration (if available)
        if self.hardware_acceleration and quality_profile.get('use_hardware_accel', True):
            hw_accel = self.hardware_acceleration
            if hw_accel == 'videotoolbox':  # macOS
                cmd.extend(['-hwaccel', 'videotoolbox'])
            elif hw_accel == 'cuda':  # NVIDIA
                cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            elif hw_accel == 'vaapi':  # Intel/AMD Linux
                cmd.extend(['-hwaccel', 'vaapi'])
        
        # Input file
        cmd.extend(['-i', str(input_path)])
        
        # Filter complex
        cmd.extend(['-filter_complex', filter_complex])
        
        # Audio stream (copy without processing)
        cmd.extend(['-c:a', 'copy'])
        
        # Video encoding settings
        codec = quality_profile.get('codec', 'libx264')
        cmd.extend(['-c:v', codec])
        
        # Quality settings
        if 'crf' in quality_profile:
            cmd.extend(['-crf', str(quality_profile['crf'])])
        
        if 'preset' in quality_profile:
            cmd.extend(['-preset', quality_profile['preset']])
        
        # Additional encoding options
        if codec == 'libx264':
            cmd.extend([
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p'  # Ensure compatibility
            ])
        
        # Output options
        cmd.extend([
            '-movflags', '+faststart',  # Enable web streaming
            '-y',  # Overwrite output
            str(output_path)
        ])
        
        return cmd
    
    def execute_with_progress(self, cmd: List[str], 
                            progress_callback: Optional[Callable] = None) -> ExecutionResult:
        """Execute FFmpeg command with progress monitoring"""
        
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        stderr_lines = []
        warning_lines = []
        
        # Monitor progress through stderr
        while True:
            line = process.stderr.readline()
            if not line:
                break
                
            stderr_lines.append(line.strip())
            
            # Parse progress information
            if progress_callback:
                progress = self._parse_progress_line(line)
                if progress is not None:
                    progress_callback(progress)
            
            # Collect warnings
            if 'warning' in line.lower():
                warning_lines.append(line.strip())
        
        # Wait for completion
        return_code = process.wait()
        execution_time = time.time() - start_time
        
        logger.info(f"FFmpeg execution completed in {execution_time:.2f}s with return code {return_code}")
        
        return ExecutionResult(
            returncode=return_code,
            execution_time=execution_time,
            stderr_lines=stderr_lines,
            warning_lines=warning_lines
        )
    
    def _parse_progress_line(self, line: str) -> Optional[float]:
        """Parse progress from FFmpeg stderr line"""
        
        # Look for time progress: frame=1234 fps=30 q=23.0 time=00:01:23.45
        if 'time=' in line:
            try:
                # Extract time portion
                time_match = re.search(r'time=(\d+):(\d+):(\d+)\.(\d+)', line)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = int(time_match.group(3))
                    centiseconds = int(time_match.group(4))
                    
                    total_seconds = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
                    return total_seconds
            except (ValueError, AttributeError):
                pass
        
        return None
```

### FilterChainBuilder

**Purpose**: Build complex FFmpeg filter graphs from timeline data.

```python
class FilterChainBuilder:
    """Build FFmpeg filter complex strings from timeline data"""
    
    def __init__(self):
        self.blur_filter_registry = {
            'gaussian': GaussianBlur,
            'pixelate': PixelateBlur,
            'noise': NoiseOverlay,
            'composite': CompositeBlur,
            'motion': MotionBlur
        }
    
    def build_filter_chain(self, timeline: Timeline, 
                          quality_profile: Dict) -> str:
        """Build complete filter complex string"""
        
        if not timeline.blur_regions:
            return "[0:v]copy[out]"
        
        # Get video info for filter optimization
        video_info = VideoInfo(
            width=quality_profile.get('width', 1920),
            height=quality_profile.get('height', 1080),
            fps=timeline.frame_rate,
            duration=timeline.video_duration
        )
        
        # Group overlapping regions for optimization
        region_groups = self._group_overlapping_regions(timeline.blur_regions)
        
        filter_parts = []
        overlay_chain = "[0:v]"
        
        for group_idx, region_group in enumerate(region_groups):
            if len(region_group) == 1:
                # Single region
                region = region_group[0]
                filter_str = self._build_single_region_filter(
                    region, group_idx, video_info
                )
                filter_parts.append(filter_str)
                
                # Add to overlay chain
                time_condition = self._build_time_condition(region)
                overlay_chain += f"[blur{group_idx}]overlay={region.bounding_box.x}:{region.bounding_box.y}:{time_condition}"
                
            else:
                # Multiple overlapping regions - merge them
                merged_filter = self._build_merged_region_filter(
                    region_group, group_idx, video_info
                )
                filter_parts.append(merged_filter)
                
                # Calculate merged region bounds
                merged_bounds = self._calculate_merged_bounds(region_group)
                time_condition = self._build_merged_time_condition(region_group)
                overlay_chain += f"[blur{group_idx}]overlay={merged_bounds.x}:{merged_bounds.y}:{time_condition}"
            
            # Chain for next overlay
            if group_idx < len(region_groups) - 1:
                overlay_chain += f"[tmp{group_idx}];[tmp{group_idx}]"
            else:
                overlay_chain += "[out]"
        
        # Combine filter parts and overlay chain
        complete_filter = ";".join(filter_parts) + ";" + overlay_chain
        
        logger.debug(f"Generated filter complex: {complete_filter}")
        return complete_filter
    
    def _build_single_region_filter(self, region: BlurRegion, index: int, 
                                   video_info: VideoInfo) -> str:
        """Build filter for single blur region"""
        
        # Get blur filter
        blur_filter = self._get_blur_filter(region.blur_type)
        
        # Crop region
        bbox = region.bounding_box
        crop_filter = f"[0:v]crop={bbox.width}:{bbox.height}:{bbox.x}:{bbox.y}"
        
        # Apply blur
        blur_filter_str = blur_filter.get_ffmpeg_filter(region, video_info)
        
        return f"{crop_filter},{blur_filter_str}[blur{index}]"
    
    def _build_time_condition(self, region: BlurRegion) -> str:
        """Build time-based enable condition"""
        return f"enable='between(t,{region.start_time},{region.end_time})'"
    
    def _get_blur_filter(self, blur_type: str) -> BaseBlurFilter:
        """Get blur filter instance by type"""
        
        if blur_type not in self.blur_filter_registry:
            logger.warning(f"Unknown blur type: {blur_type}, using gaussian")
            blur_type = 'gaussian'
        
        filter_class = self.blur_filter_registry[blur_type]
        return filter_class()
```

## Quality Management

### QualityManager

**Purpose**: Manage quality profiles and adaptive quality selection.

```python
class QualityManager:
    """Manage video quality profiles and optimization"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.profiles = self._load_quality_profiles()
        
    def select_profile(self, video_path: Path, timeline: Timeline, 
                      quality_mode: str) -> Dict:
        """Select optimal quality profile based on input and requirements"""
        
        # Get video information
        video_info = self._analyze_video(video_path)
        
        # Start with base profile
        base_profile = self.profiles[quality_mode].copy()
        
        # Adapt based on video characteristics
        adapted_profile = self._adapt_profile_to_video(base_profile, video_info)
        
        # Optimize for timeline complexity
        optimized_profile = self._optimize_for_timeline(adapted_profile, timeline)
        
        logger.info(f"Selected quality profile: {quality_mode} (adapted)")
        logger.debug(f"Profile details: {optimized_profile}")
        
        return optimized_profile
    
    def _load_quality_profiles(self) -> Dict[str, Dict]:
        """Load predefined quality profiles"""
        
        return {
            'fast': {
                'codec': 'libx264',
                'crf': 28,
                'preset': 'ultrafast',
                'use_hardware_accel': True,
                'max_resolution': (1920, 1080),
                'frame_skip': 2  # Process every 2nd frame for speed
            },
            'balanced': {
                'codec': 'libx264',
                'crf': 23,
                'preset': 'medium',
                'use_hardware_accel': True,
                'max_resolution': (3840, 2160),
                'frame_skip': 1
            },
            'high': {
                'codec': 'libx264',
                'crf': 18,
                'preset': 'slow',
                'use_hardware_accel': False,  # CPU encoding for quality
                'max_resolution': (3840, 2160),
                'frame_skip': 1
            },
            'maximum': {
                'codec': 'libx264',
                'crf': 15,
                'preset': 'veryslow',
                'use_hardware_accel': False,
                'max_resolution': (7680, 4320),  # 8K support
                'frame_skip': 1,
                'additional_options': ['-tune', 'film']
            }
        }
    
    def _adapt_profile_to_video(self, profile: Dict, video_info: VideoInfo) -> Dict:
        """Adapt profile based on video characteristics"""
        
        adapted = profile.copy()
        
        # Resolution adaptation
        if video_info.width > profile.get('max_resolution', (1920, 1080))[0]:
            # Scale down for performance
            scale_factor = profile['max_resolution'][0] / video_info.width
            adapted['scale_filter'] = f"scale={int(video_info.width * scale_factor)}:-1"
            logger.info(f"Video will be scaled down by factor {scale_factor:.2f}")
        
        # Frame rate adaptation
        if video_info.fps > 60 and profile.get('frame_skip', 1) == 1:
            # Reduce frame rate for high FPS content
            adapted['fps_filter'] = "fps=30"
            logger.info("High frame rate video will be reduced to 30 FPS")
        
        # Codec adaptation based on content
        if video_info.has_transparency and adapted['codec'] == 'libx264':
            adapted['codec'] = 'libx264'
            adapted['pix_fmt'] = 'yuva420p'  # Support alpha channel
        
        return adapted
    
    def _optimize_for_timeline(self, profile: Dict, timeline: Timeline) -> Dict:
        """Optimize profile based on timeline complexity"""
        
        optimized = profile.copy()
        
        # Calculate complexity metrics
        total_blur_regions = len(timeline.blur_regions)
        total_blur_duration = timeline.total_blur_duration()
        avg_region_size = self._calculate_avg_region_size(timeline.blur_regions)
        
        complexity_score = (
            (total_blur_regions / 10) * 0.4 +
            (total_blur_duration / timeline.video_duration) * 0.4 +
            (avg_region_size / (1920 * 1080)) * 0.2
        )
        
        # Adjust settings based on complexity
        if complexity_score > 0.8:  # High complexity
            # Reduce quality slightly for performance
            if 'crf' in optimized:
                optimized['crf'] = min(optimized['crf'] + 2, 28)
            
            # Use faster preset
            preset_map = {
                'veryslow': 'slow',
                'slow': 'medium', 
                'medium': 'fast',
                'fast': 'faster'
            }
            if optimized.get('preset') in preset_map:
                optimized['preset'] = preset_map[optimized['preset']]
            
            logger.info(f"High timeline complexity ({complexity_score:.2f}) - adjusted for performance")
        
        return optimized
```

## Progress Tracking

### ProgressTracker

**Purpose**: Monitor and report processing progress.

```python
class ProgressTracker:
    """Track and report video processing progress"""
    
    def __init__(self):
        self.current_progress = 0.0
        self.start_time = None
        self.callbacks = []
        self.stage_info = {}
        
    def start_tracking(self, total_duration: float) -> None:
        """Initialize progress tracking"""
        self.start_time = time.time()
        self.total_duration = total_duration
        self.current_progress = 0.0
        
    def update_progress(self, processed_seconds: float) -> None:
        """Update progress based on processed video duration"""
        
        if self.total_duration > 0:
            progress_percent = min((processed_seconds / self.total_duration) * 100, 100.0)
            self.current_progress = progress_percent
            
            # Calculate ETA
            if self.start_time and progress_percent > 0:
                elapsed = time.time() - self.start_time
                eta = (elapsed / (progress_percent / 100)) - elapsed
                
                progress_info = {
                    'progress_percent': progress_percent,
                    'processed_seconds': processed_seconds,
                    'total_seconds': self.total_duration,
                    'elapsed_time': elapsed,
                    'estimated_remaining': eta,
                    'processing_speed': processed_seconds / elapsed if elapsed > 0 else 0
                }
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(progress_info)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add progress callback"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove progress callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def set_stage_info(self, stage: str, info: str) -> None:
        """Set information about current processing stage"""
        self.stage_info[stage] = info
        logger.info(f"Processing stage: {stage} - {info}")
```

## Error Handling and Recovery

### ProcessingError Handling

```python
class ProcessingError(Exception):
    """Custom exception for processing errors"""
    
    def __init__(self, message: str, error_code: str = "PROCESSING_ERROR",
                 recoverable: bool = False, retry_suggestion: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable
        self.retry_suggestion = retry_suggestion

class ErrorRecoveryManager:
    """Handle processing errors and recovery strategies"""
    
    def __init__(self):
        self.retry_strategies = {
            'INSUFFICIENT_MEMORY': self._handle_memory_error,
            'CODEC_ERROR': self._handle_codec_error,
            'HARDWARE_ACCEL_FAILED': self._handle_hardware_error,
            'DISK_SPACE_ERROR': self._handle_disk_space_error
        }
    
    def handle_error(self, error: ProcessingError, 
                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle processing error and suggest recovery"""
        
        if error.error_code in self.retry_strategies:
            return self.retry_strategies[error.error_code](error, context)
        
        return None
    
    def _handle_memory_error(self, error: ProcessingError, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle out of memory errors"""
        
        return {
            'strategy': 'reduce_memory_usage',
            'modifications': {
                'chunk_duration': context.get('chunk_duration', 300) // 2,
                'quality_profile': 'fast',
                'scale_factor': 0.5
            },
            'message': 'Reducing memory usage by processing smaller chunks and lower resolution'
        }
    
    def _handle_codec_error(self, error: ProcessingError,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle codec-related errors"""
        
        return {
            'strategy': 'fallback_codec',
            'modifications': {
                'codec': 'libx264',  # Fallback to widely supported codec
                'use_hardware_accel': False
            },
            'message': 'Falling back to software encoding with libx264'
        }
```

This comprehensive processing module design provides robust, efficient, and flexible video processing capabilities while ensuring maximum privacy protection through sophisticated blur filters and quality management.