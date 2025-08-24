# Web Module Design

## Overview

The `web` module provides a modern, responsive web interface and RESTful API for the Video Privacy Editor system. It enables users to annotate video frames, review detection results, manage timelines, and monitor processing jobs through an intuitive web-based interface.

## Design Principles

1. **User-Centric Design**: Intuitive interface focused on annotation and review workflows
2. **Responsive Architecture**: Works seamlessly on desktop, tablet, and mobile devices
3. **Real-time Updates**: WebSocket integration for live progress updates
4. **RESTful API**: Clean, well-documented API following REST principles
5. **Security First**: Authentication, authorization, and input validation
6. **Performance Optimized**: Lazy loading, caching, and efficient data transfer
7. **Accessibility**: WCAG 2.1 AA compliance for inclusive design

## Module Structure

```
web/
├── __init__.py
├── DESIGN.md
├── app.py              # FastAPI application setup
├── api/                # REST API endpoints
│   ├── __init__.py
│   ├── videos.py       # Video management endpoints
│   ├── annotations.py  # Annotation CRUD endpoints
│   ├── detections.py   # Detection management endpoints
│   ├── timelines.py    # Timeline operations endpoints
│   ├── processing.py   # Video processing endpoints
│   └── system.py       # System status and health endpoints
├── models/             # Pydantic models for API
│   ├── __init__.py
│   ├── video.py        # Video-related models
│   ├── annotation.py   # Annotation models
│   ├── timeline.py     # Timeline models
│   └── common.py       # Common response models
├── middleware/         # Custom middleware
│   ├── __init__.py
│   ├── auth.py         # Authentication middleware
│   ├── cors.py         # CORS configuration
│   ├── logging.py      # Request logging
│   └── rate_limiting.py # Rate limiting
├── static/             # Static web assets
│   ├── css/
│   ├── js/
│   ├── images/
│   └── fonts/
├── templates/          # Jinja2 templates
│   ├── base.html       # Base template
│   ├── index.html      # Main application
│   ├── annotate.html   # Annotation interface
│   ├── review.html     # Review interface
│   └── timeline.html   # Timeline editor
├── websocket/          # WebSocket handlers
│   ├── __init__.py
│   ├── progress.py     # Progress updates
│   └── notifications.py # System notifications
└── frontend/           # React/Vue frontend (optional)
    ├── src/
    ├── components/
    └── pages/
```

## Architecture Overview

### Multi-Layer Web Architecture

```
Frontend (React/Vue/Vanilla JS)
       ↓
REST API Layer (FastAPI)
       ↓
Business Logic Layer (Core Modules)
       ↓
Data Layer (Storage Module)
```

### Real-time Communication

```
WebSocket Connections ←→ Progress Updates
       ↓                      ↓
Client Updates      ←→  Server Events
       ↓                      ↓
UI State Changes   ←→  Background Jobs
```

## API Design

### FastAPI Application Setup

**Purpose**: Configure FastAPI application with all necessary middleware and routes.

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Video Privacy Editor API")
    
    # Initialize storage
    storage_manager.initialize()
    
    # Start background task manager
    background_task_manager.start()
    
    # Check system dependencies
    deps = SystemUtils.check_dependencies()
    missing_deps = [name for name, info in deps.items() 
                   if info['required'] and not info['available']]
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        raise RuntimeError(f"Missing dependencies: {missing_deps}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video Privacy Editor API")
    background_task_manager.stop()
    storage_manager.close()

def create_app(config: WebConfig) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Video Privacy Editor API",
        version="1.0.0",
        description="API for video annotation, detection, and privacy processing",
        lifespan=lifespan,
        docs_url="/api/docs" if config.enable_docs else None,
        redoc_url="/api/redoc" if config.enable_docs else None
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=config.allowed_hosts
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitingMiddleware, 
                      requests_per_minute=config.rate_limit_per_minute)
    
    # Include API routers
    app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
    app.include_router(annotations.router, prefix="/api/v1/annotations", tags=["annotations"])
    app.include_router(detections.router, prefix="/api/v1/detections", tags=["detections"])
    app.include_router(timelines.router, prefix="/api/v1/timelines", tags=["timelines"])
    app.include_router(processing.router, prefix="/api/v1/processing", tags=["processing"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    
    # WebSocket routes
    app.include_router(websocket.router, prefix="/ws")
    
    # Static files and templates
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # HTML routes for web interface
    templates = Jinja2Templates(directory="templates")
    
    @app.get("/")
    async def root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
    
    @app.get("/annotate")
    async def annotate_page(request: Request):
        return templates.TemplateResponse("annotate.html", {"request": request})
    
    @app.get("/review")
    async def review_page(request: Request):
        return templates.TemplateResponse("review.html", {"request": request})
    
    @app.get("/timeline")
    async def timeline_page(request: Request):
        return templates.TemplateResponse("timeline.html", {"request": request})
    
    return app
```

### API Endpoints

#### Video Management API

**Purpose**: Handle video upload, metadata, and file operations.

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
import shutil

router = APIRouter()

@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Upload video file for processing"""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size
    if file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate video file
        validation_result = VideoUtils.validate_video_file(upload_path)
        if not validation_result.is_valid:
            upload_path.unlink()
            raise HTTPException(status_code=400, 
                              detail=f"Invalid video file: {validation_result.errors}")
        
        # Extract video metadata
        video_info = VideoUtils.get_video_info(upload_path)
        
        # Store in database
        video_record = VideoRecord(
            id=video_id,
            filename=file.filename,
            filepath=upload_path,
            duration=video_info.duration,
            frame_rate=video_info.fps,
            width=video_info.width,
            height=video_info.height,
            file_size=video_info.file_size,
            codec=video_info.codec
        )
        
        storage.save_video(video_record)
        
        # Schedule background processing if configured
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_video, video_id, upload_path
            )
        
        logger.info(f"Video uploaded successfully: {video_id}")
        
        return VideoResponse(
            id=video_id,
            filename=file.filename,
            duration=video_info.duration,
            frame_rate=video_info.fps,
            resolution={"width": video_info.width, "height": video_info.height},
            file_size=video_info.file_size,
            upload_status="completed"
        )
        
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Video upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload processing failed")

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video_info(
    video_id: str,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Get video information and metadata"""
    
    video = storage.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return VideoResponse.from_video_record(video)

@router.get("/{video_id}/frames", response_model=List[FrameResponse])
async def extract_frames(
    video_id: str,
    timestamps: List[float] = Query(...),
    quality: str = Query("high", regex="^(low|medium|high|lossless)$"),
    storage: AnnotationStorage = Depends(get_storage)
):
    """Extract frames at specified timestamps"""
    
    video = storage.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Extract frames
            frame_paths = VideoUtils.extract_frames_at_timestamps(
                video.filepath, timestamps, temp_path, quality
            )
            
            # Convert to base64 for API response
            frames = []
            for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
                if frame_path.exists():
                    # Read and encode frame
                    with open(frame_path, 'rb') as f:
                        frame_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    frames.append(FrameResponse(
                        frame_number=int(timestamp * video.frame_rate),
                        timestamp=timestamp,
                        image_data=f"data:image/png;base64,{frame_data}",
                        width=video.width,
                        height=video.height
                    ))
            
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed for video {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Frame extraction failed")

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Delete video and all associated data"""
    
    video = storage.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Delete associated annotations
        annotations = storage.get_annotations_for_video(video_id)
        for annotation in annotations:
            storage.delete_annotation(annotation.id)
        
        # Delete timelines
        timelines = storage.get_timelines_for_video(video_id)
        for timeline in timelines:
            storage.delete_timeline(timeline.id)
        
        # Delete video file
        if video.filepath.exists():
            video.filepath.unlink()
        
        # Delete video record
        storage.delete_video(video_id)
        
        logger.info(f"Video deleted: {video_id}")
        return {"message": "Video deleted successfully"}
        
    except Exception as e:
        logger.error(f"Video deletion failed for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Video deletion failed")
```

#### Annotation Management API

**Purpose**: Handle annotation creation, editing, and management.

```python
router = APIRouter()

@router.post("/", response_model=AnnotationResponse)
async def create_annotation(
    annotation_data: AnnotationCreate,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Create new annotation"""
    
    try:
        # Validate annotation data
        validation_result = ValidationUtils.validate_bounding_box(
            annotation_data.bounding_box,
            frame_size=(annotation_data.frame_width, annotation_data.frame_height)
        )
        
        if not validation_result.is_valid:
            raise HTTPException(status_code=400, 
                              detail=f"Invalid annotation: {validation_result.errors}")
        
        # Create SensitiveArea object
        annotation = SensitiveArea(
            id=str(uuid.uuid4()),
            timestamp=annotation_data.timestamp,
            bounding_box=annotation_data.bounding_box,
            area_type=annotation_data.area_type,
            confidence=1.0,  # Manual annotation
            metadata={
                'video_id': annotation_data.video_id,
                'annotated_by': annotation_data.annotated_by,
                'frame_number': int(annotation_data.timestamp * annotation_data.frame_rate)
            }
        )
        
        # Save annotation
        annotation_id = storage.save_annotation(annotation)
        
        logger.info(f"Annotation created: {annotation_id}")
        
        return AnnotationResponse.from_sensitive_area(annotation)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Annotation creation failed: {e}")
        raise HTTPException(status_code=500, detail="Annotation creation failed")

@router.get("/video/{video_id}", response_model=List[AnnotationResponse])
async def get_video_annotations(
    video_id: str,
    area_type: Optional[str] = Query(None),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
    storage: AnnotationStorage = Depends(get_storage)
):
    """Get all annotations for a video"""
    
    try:
        if start_time is not None and end_time is not None:
            # Get annotations in time range
            annotations = storage.get_annotations_in_time_range(
                video_id, start_time, end_time
            )
        else:
            # Get all annotations for video
            annotations = storage.get_annotations_for_video(video_id, area_type)
        
        return [AnnotationResponse.from_sensitive_area(ann) for ann in annotations]
        
    except Exception as e:
        logger.error(f"Failed to get annotations for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve annotations")

@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: str,
    annotation_update: AnnotationUpdate,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Update existing annotation"""
    
    try:
        # Load existing annotation
        existing = storage.load_annotation(annotation_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Update fields
        updated_annotation = existing.copy()
        
        if annotation_update.bounding_box:
            updated_annotation.bounding_box = annotation_update.bounding_box
        
        if annotation_update.area_type:
            updated_annotation.area_type = annotation_update.area_type
        
        if annotation_update.confidence is not None:
            updated_annotation.confidence = annotation_update.confidence
        
        # Update metadata
        updated_annotation.metadata.update(annotation_update.metadata or {})
        updated_annotation.metadata['updated_at'] = datetime.utcnow().isoformat()
        
        # Validate updated annotation
        if updated_annotation.bounding_box:
            validation_result = ValidationUtils.validate_bounding_box(
                updated_annotation.bounding_box
            )
            
            if not validation_result.is_valid:
                raise HTTPException(status_code=400, 
                                  detail=f"Invalid annotation: {validation_result.errors}")
        
        # Save updated annotation
        storage.save_annotation(updated_annotation)
        
        logger.info(f"Annotation updated: {annotation_id}")
        
        return AnnotationResponse.from_sensitive_area(updated_annotation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Annotation update failed for {annotation_id}: {e}")
        raise HTTPException(status_code=500, detail="Annotation update failed")

@router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: str,
    storage: AnnotationStorage = Depends(get_storage)
):
    """Delete annotation"""
    
    try:
        success = storage.delete_annotation(annotation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        logger.info(f"Annotation deleted: {annotation_id}")
        return {"message": "Annotation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Annotation deletion failed for {annotation_id}: {e}")
        raise HTTPException(status_code=500, detail="Annotation deletion failed")
```

## Frontend Interface Design

### Annotation Interface

**Purpose**: Interactive video annotation tool for drawing bounding boxes.

**Key Features**:
- **Video Player**: Custom HTML5 player with frame-by-frame navigation
- **Drawing Tools**: Precise bounding box drawing with mouse/touch
- **Keyboard Shortcuts**: Efficiency shortcuts for rapid annotation
- **Annotation Management**: Edit, delete, and categorize annotations
- **Progress Tracking**: Visual progress through annotation tasks

```javascript
class AnnotationInterface {
    constructor(containerId, videoUrl) {
        this.container = document.getElementById(containerId);
        this.videoUrl = videoUrl;
        this.annotations = new Map();
        this.currentAnnotation = null;
        this.isDrawing = false;
        
        this.setupInterface();
        this.setupEventListeners();
        this.loadVideo();
    }
    
    setupInterface() {
        this.container.innerHTML = `
            <div class="annotation-workspace">
                <div class="video-panel">
                    <div class="video-container">
                        <video id="annotationVideo" controls>
                            <source src="${this.videoUrl}" type="video/mp4">
                        </video>
                        <canvas id="annotationCanvas" class="annotation-overlay"></canvas>
                    </div>
                    
                    <div class="video-controls">
                        <button id="prevFrame" class="control-btn">◀ Frame</button>
                        <button id="playPause" class="control-btn">⏯ Play</button>
                        <button id="nextFrame" class="control-btn">Frame ▶</button>
                        <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="100" value="0">
                        <span id="timeDisplay">00:00 / 00:00</span>
                    </div>
                </div>
                
                <div class="annotation-panel">
                    <div class="tools-section">
                        <h3>Annotation Tools</h3>
                        <div class="tool-buttons">
                            <button id="drawTool" class="tool-btn active" data-tool="draw">Draw Box</button>
                            <button id="selectTool" class="tool-btn" data-tool="select">Select</button>
                            <button id="deleteTool" class="tool-btn" data-tool="delete">Delete</button>
                        </div>
                        
                        <div class="area-type-selector">
                            <label for="areaType">Area Type:</label>
                            <select id="areaType">
                                <option value="chatgpt">ChatGPT</option>
                                <option value="atuin">Atuin Terminal</option>
                                <option value="terminal">Generic Terminal</option>
                                <option value="custom">Custom</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="annotations-list">
                        <h3>Annotations</h3>
                        <div id="annotationsList" class="annotations-container">
                            <!-- Annotation items will be populated here -->
                        </div>
                    </div>
                    
                    <div class="actions-section">
                        <button id="saveAnnotations" class="action-btn primary">Save Annotations</button>
                        <button id="exportAnnotations" class="action-btn secondary">Export</button>
                        <button id="clearAll" class="action-btn danger">Clear All</button>
                    </div>
                </div>
            </div>
        `;
    }
    
    setupEventListeners() {
        const video = document.getElementById('annotationVideo');
        const canvas = document.getElementById('annotationCanvas');
        const ctx = canvas.getContext('2d');
        
        // Video events
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            this.updateTimeDisplay();
        });
        
        video.addEventListener('timeupdate', () => {
            this.updateTimeDisplay();
            this.updateTimelineSlider();
            this.renderAnnotations();
        });
        
        // Canvas drawing events
        canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        canvas.addEventListener('mousemove', (e) => this.continueDrawing(e));
        canvas.addEventListener('mouseup', (e) => this.finishDrawing(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.stepFrame(-1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.stepFrame(1);
                    break;
                case ' ':
                    e.preventDefault();
                    this.togglePlayPause();
                    break;
                case 'Delete':
                case 'Backspace':
                    this.deleteSelectedAnnotation();
                    break;
            }
        });
        
        // Control buttons
        document.getElementById('prevFrame').onclick = () => this.stepFrame(-1);
        document.getElementById('nextFrame').onclick = () => this.stepFrame(1);
        document.getElementById('playPause').onclick = () => this.togglePlayPause();
        
        // Timeline slider
        document.getElementById('timelineSlider').oninput = (e) => {
            const video = document.getElementById('annotationVideo');
            const percentage = e.target.value / 100;
            video.currentTime = video.duration * percentage;
        };
        
        // Tool buttons
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.onclick = () => this.selectTool(btn.dataset.tool);
        });
        
        // Action buttons
        document.getElementById('saveAnnotations').onclick = () => this.saveAnnotations();
        document.getElementById('exportAnnotations').onclick = () => this.exportAnnotations();
        document.getElementById('clearAll').onclick = () => this.clearAllAnnotations();
    }
    
    startDrawing(e) {
        if (this.currentTool !== 'draw') return;
        
        this.isDrawing = true;
        const rect = e.target.getBoundingClientRect();
        const scaleX = e.target.width / rect.width;
        const scaleY = e.target.height / rect.height;
        
        this.drawStartX = (e.clientX - rect.left) * scaleX;
        this.drawStartY = (e.clientY - rect.top) * scaleY;
        
        this.currentAnnotation = {
            id: this.generateId(),
            timestamp: document.getElementById('annotationVideo').currentTime,
            x: this.drawStartX,
            y: this.drawStartY,
            width: 0,
            height: 0,
            areaType: document.getElementById('areaType').value,
            confidence: 1.0
        };
    }
    
    continueDrawing(e) {
        if (!this.isDrawing || !this.currentAnnotation) return;
        
        const rect = e.target.getBoundingClientRect();
        const scaleX = e.target.width / rect.width;
        const scaleY = e.target.height / rect.height;
        
        const currentX = (e.clientX - rect.left) * scaleX;
        const currentY = (e.clientY - rect.top) * scaleY;
        
        this.currentAnnotation.width = currentX - this.drawStartX;
        this.currentAnnotation.height = currentY - this.drawStartY;
        
        this.renderAnnotations();
    }
    
    finishDrawing(e) {
        if (!this.isDrawing || !this.currentAnnotation) return;
        
        this.isDrawing = false;
        
        // Normalize negative dimensions
        if (this.currentAnnotation.width < 0) {
            this.currentAnnotation.x += this.currentAnnotation.width;
            this.currentAnnotation.width = Math.abs(this.currentAnnotation.width);
        }
        
        if (this.currentAnnotation.height < 0) {
            this.currentAnnotation.y += this.currentAnnotation.height;
            this.currentAnnotation.height = Math.abs(this.currentAnnotation.height);
        }
        
        // Only save if annotation is meaningful size
        if (this.currentAnnotation.width > 10 && this.currentAnnotation.height > 10) {
            this.annotations.set(this.currentAnnotation.id, this.currentAnnotation);
            this.updateAnnotationsList();
        }
        
        this.currentAnnotation = null;
        this.renderAnnotations();
    }
    
    renderAnnotations() {
        const canvas = document.getElementById('annotationCanvas');
        const ctx = canvas.getContext('2d');
        const currentTime = document.getElementById('annotationVideo').currentTime;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Render existing annotations near current time
        const timeWindow = 0.1; // Show annotations within 100ms
        
        this.annotations.forEach(annotation => {
            if (Math.abs(annotation.timestamp - currentTime) <= timeWindow) {
                this.drawAnnotationBox(ctx, annotation, 'rgba(0, 255, 0, 0.3)', 'rgba(0, 255, 0, 1)');
            }
        });
        
        // Render current drawing annotation
        if (this.currentAnnotation && this.isDrawing) {
            this.drawAnnotationBox(ctx, this.currentAnnotation, 'rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 0, 1)');
        }
    }
    
    drawAnnotationBox(ctx, annotation, fillColor, strokeColor) {
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        
        ctx.fillRect(annotation.x, annotation.y, annotation.width, annotation.height);
        ctx.strokeRect(annotation.x, annotation.y, annotation.width, annotation.height);
        
        // Draw label
        ctx.fillStyle = strokeColor;
        ctx.font = '14px Arial';
        ctx.fillText(
            annotation.areaType, 
            annotation.x, 
            annotation.y - 5
        );
    }
    
    async saveAnnotations() {
        const annotationsArray = Array.from(this.annotations.values());
        
        try {
            const response = await fetch('/api/v1/annotations/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_id: this.videoId,
                    annotations: annotationsArray.map(ann => ({
                        timestamp: ann.timestamp,
                        bounding_box: {
                            x: Math.round(ann.x),
                            y: Math.round(ann.y),
                            width: Math.round(ann.width),
                            height: Math.round(ann.height)
                        },
                        area_type: ann.areaType,
                        confidence: ann.confidence
                    }))
                })
            });
            
            if (response.ok) {
                this.showNotification('Annotations saved successfully', 'success');
            } else {
                const error = await response.json();
                this.showNotification(`Save failed: ${error.detail}`, 'error');
            }
            
        } catch (error) {
            this.showNotification(`Save failed: ${error.message}`, 'error');
        }
    }
}
```

### Timeline Editor Interface

**Purpose**: Visual timeline editor for reviewing and adjusting blur regions.

```javascript
class TimelineEditor {
    constructor(containerId, timelineData) {
        this.container = document.getElementById(containerId);
        this.timeline = timelineData;
        this.scale = 100; // pixels per second
        this.selectedRegion = null;
        
        this.setupInterface();
        this.setupEventListeners();
        this.renderTimeline();
    }
    
    setupInterface() {
        this.container.innerHTML = `
            <div class="timeline-editor">
                <div class="timeline-header">
                    <h2>Blur Timeline Editor</h2>
                    <div class="timeline-controls">
                        <button id="zoomIn" class="control-btn">🔍+</button>
                        <button id="zoomOut" class="control-btn">🔍-</button>
                        <button id="playPreview" class="control-btn">▶ Preview</button>
                        <span class="time-display" id="currentTime">00:00</span>
                    </div>
                </div>
                
                <div class="timeline-workspace">
                    <div class="timeline-ruler" id="timelineRuler"></div>
                    <div class="timeline-tracks" id="timelineTracks">
                        <!-- Timeline tracks will be rendered here -->
                    </div>
                    <div class="timeline-playhead" id="playhead"></div>
                </div>
                
                <div class="timeline-properties" id="regionProperties">
                    <h3>Region Properties</h3>
                    <div class="property-group">
                        <label>Blur Type:</label>
                        <select id="blurType">
                            <option value="gaussian">Gaussian</option>
                            <option value="pixelate">Pixelate</option>
                            <option value="noise">Noise</option>
                            <option value="composite">Composite</option>
                        </select>
                    </div>
                    
                    <div class="property-group">
                        <label>Blur Strength:</label>
                        <input type="range" id="blurStrength" min="0.1" max="2.0" step="0.1" value="1.0">
                        <span id="strengthValue">1.0</span>
                    </div>
                    
                    <div class="property-group">
                        <label>Start Time:</label>
                        <input type="number" id="startTime" step="0.1" min="0">
                    </div>
                    
                    <div class="property-group">
                        <label>End Time:</label>
                        <input type="number" id="endTime" step="0.1" min="0">
                    </div>
                    
                    <div class="property-actions">
                        <button id="applyChanges" class="action-btn primary">Apply Changes</button>
                        <button id="deleteRegion" class="action-btn danger">Delete Region</button>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderTimeline() {
        this.renderRuler();
        this.renderTracks();
        this.renderRegions();
    }
    
    renderRuler() {
        const ruler = document.getElementById('timelineRuler');
        ruler.innerHTML = '';
        
        const duration = this.timeline.video_duration;
        const width = duration * this.scale;
        
        ruler.style.width = `${width}px`;
        
        // Add time markers
        for (let time = 0; time <= duration; time += 10) { // Every 10 seconds
            const marker = document.createElement('div');
            marker.className = 'time-marker';
            marker.style.left = `${time * this.scale}px`;
            marker.textContent = this.formatTime(time);
            ruler.appendChild(marker);
        }
    }
    
    renderRegions() {
        const tracksContainer = document.getElementById('timelineTracks');
        
        // Clear existing regions
        tracksContainer.querySelectorAll('.blur-region').forEach(el => el.remove());
        
        // Render blur regions
        this.timeline.blur_regions.forEach((region, index) => {
            const regionElement = document.createElement('div');
            regionElement.className = 'blur-region';
            regionElement.dataset.regionId = region.id;
            
            const left = region.start_time * this.scale;
            const width = (region.end_time - region.start_time) * this.scale;
            
            regionElement.style.left = `${left}px`;
            regionElement.style.width = `${width}px`;
            regionElement.style.backgroundColor = this.getBlurTypeColor(region.blur_type);
            
            // Add region content
            regionElement.innerHTML = `
                <div class="region-label">${region.blur_type}</div>
                <div class="region-handles">
                    <div class="handle handle-start"></div>
                    <div class="handle handle-end"></div>
                </div>
            `;
            
            // Add event listeners for selection and dragging
            regionElement.onclick = () => this.selectRegion(region);
            
            tracksContainer.appendChild(regionElement);
        });
    }
    
    selectRegion(region) {
        // Update visual selection
        document.querySelectorAll('.blur-region').forEach(el => 
            el.classList.remove('selected')
        );
        
        document.querySelector(`[data-region-id="${region.id}"]`)
            ?.classList.add('selected');
        
        this.selectedRegion = region;
        this.updatePropertiesPanel(region);
    }
    
    updatePropertiesPanel(region) {
        document.getElementById('blurType').value = region.blur_type;
        document.getElementById('blurStrength').value = region.blur_strength;
        document.getElementById('strengthValue').textContent = region.blur_strength;
        document.getElementById('startTime').value = region.start_time.toFixed(1);
        document.getElementById('endTime').value = region.end_time.toFixed(1);
    }
}
```

## WebSocket Integration

### Real-time Progress Updates

**Purpose**: Provide real-time progress updates for long-running operations.

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.job_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from job subscriptions
        for job_id, subscribers in self.job_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
    
    async def subscribe_to_job(self, websocket: WebSocket, job_id: str):
        if job_id not in self.job_subscribers:
            self.job_subscribers[job_id] = []
        self.job_subscribers[job_id].append(websocket)
    
    async def broadcast_job_update(self, job_id: str, message: dict):
        if job_id in self.job_subscribers:
            disconnected = []
            for connection in self.job_subscribers[job_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.job_subscribers[job_id].remove(conn)

manager = ConnectionManager()

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for job progress updates"""
    
    await manager.connect(websocket)
    await manager.subscribe_to_job(websocket, job_id)
    
    try:
        # Send initial job status
        job_status = get_job_status(job_id)
        if job_status:
            await websocket.send_text(json.dumps({
                "type": "job_status",
                "job_id": job_id,
                "status": job_status.status,
                "progress": job_status.progress
            }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Progress update function called from background tasks
async def update_job_progress(job_id: str, progress: float, message: str = None):
    """Send progress update to subscribed clients"""
    
    update_message = {
        "type": "progress_update",
        "job_id": job_id,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.broadcast_job_update(job_id, update_message)
```

## Security and Authentication

### Authentication Middleware

**Purpose**: Handle user authentication and authorization.

```python
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from passlib.context import CryptContext

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    """Handle authentication and authorization"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

auth_manager = AuthManager(SECRET_KEY)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Load user from database
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# Dependency for protected routes
async def require_auth(current_user = Depends(get_current_user)):
    return current_user
```

This comprehensive web module design provides a complete web interface and API for the Video Privacy Editor system, enabling intuitive annotation workflows, real-time progress monitoring, and secure user interactions.