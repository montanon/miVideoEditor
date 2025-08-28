/**
 * Video Annotation Tool - Frontend Application
 * Integrates with backend API for video annotation workflow
 */

class VideoAnnotationApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.currentVideoId = null;
        this.annotations = [];
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentRect = null;
        this.videoMetadata = {};
        this.videoBlob = null;
        
        // Time range annotation state
        this.isTimeRangeMode = false;
        this.timeRangeStart = null;
        this.timeRangeEnd = null;
        this.timeRangeAnnotations = [];
        
        this.init();
    }

    init() {
        this.setupElements();
        this.bindEvents();
        console.log('Video Annotation App initialized');
    }

    setupElements() {
        // Video elements
        this.video = document.getElementById('videoPlayer');
        this.canvas = document.getElementById('annotationCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Control elements
        this.videoInput = document.getElementById('videoInput');
        this.uploadSection = document.getElementById('uploadSection');
        this.videoSection = document.getElementById('videoSection');
        this.annotationsSection = document.getElementById('annotationsSection');
        
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.frameBackBtn = document.getElementById('frameBackBtn');
        this.frameForwardBtn = document.getElementById('frameForwardBtn');
        this.currentTimeDisplay = document.getElementById('currentTime');
        this.durationDisplay = document.getElementById('videoDuration');
        
        this.timeline = document.getElementById('timeline');
        this.timelineProgress = document.getElementById('timelineProgress');
        this.timelineHandle = document.getElementById('timelineHandle');
        this.timelineAnnotations = document.getElementById('timelineAnnotations');
        
        this.areaTypeSelect = document.getElementById('areaType');
        this.saveAnnotationBtn = document.getElementById('saveAnnotationBtn');
        this.clearCanvasBtn = document.getElementById('clearCanvasBtn');
        
        // Time range elements
        this.toggleTimeRangeBtn = document.getElementById('toggleTimeRangeBtn');
        this.timeRangeControls = document.getElementById('timeRangeControls');
        this.sampleInterval = document.getElementById('sampleInterval');
        this.rangeStart = document.getElementById('rangeStart');
        this.rangeEnd = document.getElementById('rangeEnd');
        this.rangeDuration = document.getElementById('rangeDuration');
        
        // Replace video elements
        this.replaceVideoBtn = document.getElementById('replaceVideoBtn');
        this.replaceVideoSection = document.getElementById('replaceVideoSection');
        this.replaceVideoInput = document.getElementById('replaceVideoInput');
        this.cancelReplaceBtn = document.getElementById('cancelReplaceBtn');
        this.drawingInstructions = document.getElementById('drawingInstructions');
        
        this.annotationsList = document.getElementById('annotationsList');
        this.annotationsStats = document.getElementById('annotationsStats');
        this.exportJsonBtn = document.getElementById('exportJsonBtn');
        this.exportCocoBtn = document.getElementById('exportCocoBtn');

        this.statusMessage = document.getElementById('statusMessage');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    bindEvents() {
        // Video upload
        this.videoInput.addEventListener('change', (e) => this.handleVideoUpload(e));

        // Video events
        this.video.addEventListener('loadedmetadata', () => this.onVideoLoaded());
        this.video.addEventListener('timeupdate', () => this.onTimeUpdate());
        this.video.addEventListener('click', () => this.togglePlayPause());

        // Canvas drawing events
        this.canvas.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));

        // Control buttons
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.frameBackBtn.addEventListener('click', () => this.seekFrames(-1));
        this.frameForwardBtn.addEventListener('click', () => this.seekFrames(1));
        this.saveAnnotationBtn.addEventListener('click', () => this.saveCurrentAnnotation());
        this.clearCanvasBtn.addEventListener('click', () => this.clearCanvas());
        
        // Time range controls
        this.toggleTimeRangeBtn.addEventListener('click', () => this.toggleTimeRangeMode());
        
        // Replace video controls
        this.replaceVideoBtn.addEventListener('click', () => this.showReplaceVideoSection());
        this.replaceVideoInput.addEventListener('change', (e) => this.handleVideoReplace(e));
        this.cancelReplaceBtn.addEventListener('click', () => this.hideReplaceVideoSection());

        // Timeline
        this.timeline.addEventListener('click', (e) => this.onTimelineClick(e));

        // Export buttons
        this.exportJsonBtn.addEventListener('click', () => this.exportAnnotations('json'));
        this.exportCocoBtn.addEventListener('click', () => this.exportAnnotations('coco'));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Window resize
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async handleVideoUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showLoading('Uploading video...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/videos/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            this.currentVideoId = result.video_id;
            this.videoMetadata = result;

            // Set video source to use frame extraction endpoint
            this.video.src = URL.createObjectURL(file);
            
            this.showVideoSection();
            this.showSuccess(`Video uploaded successfully: ${result.filename}`);

        } catch (error) {
            this.showError(`Upload failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    // Video Replacement Methods
    showReplaceVideoSection() {
        if (!this.currentVideoId) {
            this.showError('Please upload a video first');
            return;
        }
        this.replaceVideoSection.style.display = 'block';
        this.replaceVideoBtn.textContent = '⏳ Replace Mode';
        this.replaceVideoBtn.style.background = '#ef4444';
    }
    
    hideReplaceVideoSection() {
        this.replaceVideoSection.style.display = 'none';
        this.replaceVideoBtn.textContent = '🔄 Replace Video';
        this.replaceVideoBtn.style.background = '#f59e0b';
        this.replaceVideoInput.value = '';
    }
    
    async handleVideoReplace(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!this.currentVideoId) {
            this.showError('No video to replace');
            return;
        }

        const confirmed = confirm(`Are you sure you want to replace the current video? This will:\n\n✓ Keep all existing annotations\n✓ Replace the video file\n✓ Update video metadata\n\nThis cannot be undone.`);
        if (!confirmed) {
            this.replaceVideoInput.value = '';
            return;
        }

        this.showLoading('Replacing video...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`/api/videos/${this.currentVideoId}/replace`, {
                method: 'PUT',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Video replacement failed');
            }

            const result = await response.json();
            
            // Update video metadata but keep the same ID
            this.videoMetadata = result;

            // Clean up old video blob and set new source
            if (this.video.src && this.video.src.startsWith('blob:')) {
                URL.revokeObjectURL(this.video.src);
            }
            this.video.src = URL.createObjectURL(file);
            
            // Hide replace section
            this.hideReplaceVideoSection();
            
            // Clear current drawing state but keep annotations
            this.clearCanvas();
            
            // Reload annotations to ensure they're still valid
            await this.loadAnnotations();
            await this.loadTimeRangeAnnotations();
            
            this.showSuccess(`Video replaced successfully: ${result.filename}`);

        } catch (error) {
            this.showError(`Video replacement failed: ${error.message}`);
            this.replaceVideoInput.value = '';
        } finally {
            this.hideLoading();
        }
    }

    onVideoLoaded() {
        this.resizeCanvas();
        this.setupTimeline();
        this.loadAnnotations();
        
        this.durationDisplay.textContent = this.formatTime(this.video.duration);
        console.log('Video loaded:', this.videoMetadata);
    }

    onTimeUpdate() {
        const currentTime = this.video.currentTime;
        const duration = this.video.duration;
        
        this.currentTimeDisplay.textContent = this.formatTime(currentTime);
        
        // Update timeline progress
        const progress = (currentTime / duration) * 100;
        this.timelineProgress.style.width = `${progress}%`;
        this.timelineHandle.style.left = `${progress}%`;
    }

    resizeCanvas() {
        if (!this.video || !this.canvas) return;

        const rect = this.video.getBoundingClientRect();
        const pixelRatio = window.devicePixelRatio || 1;
        
        // Set actual size in memory (scaled to device pixel ratio)
        this.canvas.width = (this.video.videoWidth || rect.width) * pixelRatio;
        this.canvas.height = (this.video.videoHeight || rect.height) * pixelRatio;
        
        // Set display size (CSS pixels)
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;
        
        // Scale the context to ensure correct drawing operations
        this.ctx.scale(pixelRatio, pixelRatio);

        this.canvas.classList.add('drawing');
    }

    // Canvas Drawing Methods
    getCanvasCoordinates(event) {
        const rect = this.canvas.getBoundingClientRect();
        
        // Get mouse position relative to canvas display size
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Since the context is already scaled by pixelRatio in resizeCanvas(),
        // we just need the coordinates relative to the display size
        return { x: mouseX, y: mouseY };
    }
    
    onCanvasMouseDown(event) {
        if (this.video.paused) {
            const coords = this.getCanvasCoordinates(event);
            this.startX = coords.x;
            this.startY = coords.y;
            this.isDrawing = true;
            this.clearCanvas();
        }
    }

    onCanvasMouseMove(event) {
        if (!this.isDrawing) return;

        const coords = this.getCanvasCoordinates(event);
        this.drawBoundingBox(this.startX, this.startY, coords.x, coords.y);
    }

    onCanvasMouseUp(event) {
        if (!this.isDrawing) return;

        const coords = this.getCanvasCoordinates(event);
        const endX = coords.x;
        const endY = coords.y;

        this.isDrawing = false;

        // Calculate bounding box coordinates
        const x = Math.min(this.startX, endX);
        const y = Math.min(this.startY, endY);
        const width = Math.abs(endX - this.startX);
        const height = Math.abs(endY - this.startY);
        
        // Convert from canvas display coordinates to video pixel coordinates
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.video.videoWidth / rect.width;
        const scaleY = this.video.videoHeight / rect.height;
        
        const videoX = Math.round(x * scaleX);
        const videoY = Math.round(y * scaleY);
        const videoWidth = Math.round(width * scaleX);
        const videoHeight = Math.round(height * scaleY);

        if (videoWidth > 10 && videoHeight > 10) {
            this.currentRect = { 
                x: videoX, 
                y: videoY, 
                width: videoWidth, 
                height: videoHeight 
            };
            this.saveAnnotationBtn.disabled = false;
            this.showInfo('Bounding box created. Click "Save Annotation" to save.');
        } else {
            this.clearCanvas();
            this.showWarning('Bounding box too small. Try drawing a larger area.');
        }
    }

    drawBoundingBox(x1, y1, x2, y2) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const x = Math.min(x1, x2);
        const y = Math.min(y1, y2);
        const width = Math.abs(x2 - x1);
        const height = Math.abs(y2 - y1);

        this.ctx.strokeStyle = '#ef4444';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(x, y, width, height);

        this.ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
        this.ctx.fillRect(x, y, width, height);
    }

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.currentRect = null;
        this.saveAnnotationBtn.disabled = true;
    }

    // Annotation Management
    async saveCurrentAnnotation() {
        if (!this.currentRect || !this.currentVideoId) return;

        // Handle time range mode
        if (this.isTimeRangeMode) {
            if (this.timeRangeStart === null || this.timeRangeEnd === null) {
                this.showError('Please select both start and end times for the time range');
                return;
            }
            await this.saveTimeRangeAnnotation();
            return;
        }

        this.showLoading('Saving annotation...');

        try {
            const annotationData = {
                video_id: this.currentVideoId,
                timestamp: this.video.currentTime,
                bounding_box: this.currentRect,
                area_type: this.areaTypeSelect.value,
                confidence: 1.0
            };

            const response = await fetch('/api/annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(annotationData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save annotation');
            }

            const result = await response.json();
            this.clearCanvas();
            await this.loadAnnotations();
            
            this.showSuccess('Annotation saved successfully!');

        } catch (error) {
            this.showError(`Failed to save annotation: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async loadAnnotations() {
        if (!this.currentVideoId) return;

        try {
            const response = await fetch(`/api/annotations/video/${this.currentVideoId}`);
            if (!response.ok) {
                throw new Error('Failed to load annotations');
            }

            this.annotations = await response.json();
            this.updateAnnotationsList();
            this.updateTimeline();
            this.updateStats();
            
            if (this.annotations.length > 0) {
                this.annotationsSection.style.display = 'block';
            }

        } catch (error) {
            this.showError(`Failed to load annotations: ${error.message}`);
        }
    }

    updateAnnotationsList() {
        this.annotationsList.innerHTML = '';

        if (this.annotations.length === 0) {
            this.annotationsList.innerHTML = '<p class="no-annotations">No annotations yet. Create your first annotation by drawing on the video.</p>';
            return;
        }

        this.annotations.forEach(annotation => {
            const item = document.createElement('div');
            item.className = 'annotation-item';
            
            const bbox = annotation.bounding_box;
            const timestamp = this.formatTime(annotation.timestamp);
            
            item.innerHTML = `
                <div class="annotation-info">
                    <div class="annotation-timestamp">${timestamp}</div>
                    <div class="annotation-type ${annotation.area_type}">${annotation.area_type.toUpperCase()}</div>
                    <div class="annotation-details">
                        ${bbox.width}×${bbox.height} px at (${bbox.x}, ${bbox.y})
                        | Confidence: ${(annotation.confidence * 100).toFixed(0)}%
                    </div>
                </div>
                <div class="annotation-actions">
                    <button class="goto-btn" onclick="app.seekToAnnotation(${annotation.timestamp})">
                        🎯 Go to
                    </button>
                    <button class="delete-btn" onclick="app.deleteAnnotation('${annotation.id}')">
                        🗑️ Delete
                    </button>
                </div>
            `;
            
            this.annotationsList.appendChild(item);
        });
    }

    updateStats() {
        const stats = this.calculateStats();
        
        this.annotationsStats.innerHTML = `
            <strong>Statistics:</strong>
            Total: ${stats.total} annotations |
            Coverage: ${stats.coverage.toFixed(1)}% |
            Types: ${Object.entries(stats.byType).map(([type, count]) => `${type}: ${count}`).join(', ')}
        `;
    }

    calculateStats() {
        const total = this.annotations.length;
        const byType = {};
        
        this.annotations.forEach(ann => {
            byType[ann.area_type] = (byType[ann.area_type] || 0) + 1;
        });

        // Calculate coverage (rough estimate based on annotations)
        const uniqueTimestamps = new Set(this.annotations.map(ann => Math.floor(ann.timestamp)));
        const coverage = this.video ? (uniqueTimestamps.size / this.video.duration) * 100 : 0;

        return { total, byType, coverage };
    }

    updateTimeline() {
        this.timelineAnnotations.innerHTML = '';

        this.annotations.forEach(annotation => {
            const span = document.createElement('div');
            span.className = `timeline-annotation ${annotation.area_type}`;
            
            const position = (annotation.timestamp / this.video.duration) * 100;
            span.style.left = `${position}%`;
            span.style.width = '2px'; // Thin line for each annotation
            
            span.title = `${this.formatTime(annotation.timestamp)} - ${annotation.area_type}`;
            
            this.timelineAnnotations.appendChild(span);
        });
    }

    seekToAnnotation(timestamp) {
        this.video.currentTime = timestamp;
        this.video.pause();
    }

    async deleteAnnotation(annotationId) {
        if (!confirm('Are you sure you want to delete this annotation?')) return;

        this.showLoading('Deleting annotation...');

        try {
            const response = await fetch(`/api/annotations/${annotationId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to delete annotation');
            }

            await this.loadAnnotations();
            this.showSuccess('Annotation deleted successfully!');

        } catch (error) {
            this.showError(`Failed to delete annotation: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    // Export functionality
    async exportAnnotations(format) {
        if (!this.currentVideoId) return;

        this.showLoading(`Exporting ${format.toUpperCase()}...`);

        try {
            const response = await fetch(`/api/annotations/video/${this.currentVideoId}/export?format=${format}`);
            
            if (!response.ok) {
                throw new Error(`Export failed: ${response.statusText}`);
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `annotations_${this.currentVideoId}_${format}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.showSuccess(`${format.toUpperCase()} export completed!`);

        } catch (error) {
            this.showError(`Export failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    // Video Controls
    togglePlayPause() {
        if (this.video.paused) {
            this.video.play();
        } else {
            this.video.pause();
        }
    }

    seekFrames(direction) {
        const frameTime = 1 / (this.videoMetadata.frame_rate || 30);
        this.video.currentTime += frameTime * direction;
    }

    onTimelineClick(event) {
        const rect = this.timeline.getBoundingClientRect();
        const position = (event.clientX - rect.left) / rect.width;
        const clickTime = position * this.video.duration;
        
        if (this.isTimeRangeMode) {
            this.setTimeRangePoint(clickTime);
        } else {
            this.video.currentTime = clickTime;
        }
    }

    setupTimeline() {
        // Timeline is set up and will be updated as annotations are loaded
    }

    // Keyboard shortcuts
    handleKeyboard(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;

        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.togglePlayPause();
                break;
            case 'ArrowLeft':
                event.preventDefault();
                this.seekFrames(-1);
                break;
            case 'ArrowRight':
                event.preventDefault();
                this.seekFrames(1);
                break;
            case 'Enter':
                if (this.currentRect) {
                    this.saveCurrentAnnotation();
                }
                break;
            case 'Escape':
                this.clearCanvas();
                break;
        }
    }

    // UI State Management
    showVideoSection() {
        this.uploadSection.style.display = 'none';
        this.videoSection.style.display = 'block';
    }

    // Utility Methods
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    showLoading(message) {
        this.loadingOverlay.querySelector('.loading-text').textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showMessage(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.classList.add('show');

        setTimeout(() => {
            this.statusMessage.classList.remove('show');
        }, 4000);
    }

    showSuccess(message) { this.showMessage(message, 'success'); }
    showError(message) { this.showMessage(message, 'error'); }
    showWarning(message) { this.showMessage(message, 'warning'); }
    showInfo(message) { this.showMessage(message, 'info'); }
    
    // Memory management
    cleanupVideo() {
        if (this.video && this.video.src && this.video.src.startsWith('blob:')) {
            URL.revokeObjectURL(this.video.src);
            this.video.src = '';
        }
        if (this.videoBlob) {
            this.videoBlob = null;
        }
    }
    
    // Cleanup on page unload
    cleanup() {
        this.cleanupVideo();
        if (this.canvas && this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        this.annotations = [];
        this.currentVideoId = null;
    }
    
    // Time Range Annotation Methods
    toggleTimeRangeMode() {
        this.isTimeRangeMode = !this.isTimeRangeMode;
        
        if (this.isTimeRangeMode) {
            this.toggleTimeRangeBtn.textContent = '🎯 Single Point Mode';
            this.toggleTimeRangeBtn.style.background = '#ef4444';
            this.timeRangeControls.style.display = 'block';
            this.drawingInstructions.textContent = 'Time Range Mode: Click timeline to set start/end times, then draw bounding box';
            this.saveAnnotationBtn.textContent = '💾 Save Time Range';
        } else {
            this.toggleTimeRangeBtn.textContent = '⏱️ Time Range Mode';
            this.toggleTimeRangeBtn.style.background = '';
            this.timeRangeControls.style.display = 'none';
            this.drawingInstructions.textContent = 'Click and drag on the video to draw bounding boxes';
            this.saveAnnotationBtn.textContent = '💾 Save Annotation';
            this.resetTimeRange();
        }
        
        this.clearCanvas();
    }
    
    resetTimeRange() {
        this.timeRangeStart = null;
        this.timeRangeEnd = null;
        this.rangeStart.textContent = '--';
        this.rangeEnd.textContent = '--';
        this.rangeDuration.textContent = '--';
    }
    
    setTimeRangePoint(timestamp) {
        if (!this.isTimeRangeMode) return;
        
        if (this.timeRangeStart === null) {
            // Set start time
            this.timeRangeStart = timestamp;
            this.rangeStart.textContent = timestamp.toFixed(1);
            this.showInfo(`Range start set to ${timestamp.toFixed(1)}s. Click timeline again to set end time.`);
        } else if (this.timeRangeEnd === null) {
            // Set end time
            if (timestamp <= this.timeRangeStart) {
                this.showWarning('End time must be after start time. Please select a later timestamp.');
                return;
            }
            
            this.timeRangeEnd = timestamp;
            this.rangeEnd.textContent = timestamp.toFixed(1);
            
            const duration = this.timeRangeEnd - this.timeRangeStart;
            this.rangeDuration.textContent = duration.toFixed(1);
            
            this.showSuccess(`Time range set: ${this.timeRangeStart.toFixed(1)}s - ${this.timeRangeEnd.toFixed(1)}s (${duration.toFixed(1)}s). Now draw a bounding box.`);
            
            // Update timeline visualization
            this.updateTimelineRangeDisplay();
        } else {
            // Reset and start over
            this.resetTimeRange();
            this.setTimeRangePoint(timestamp);
        }
    }
    
    updateTimelineRangeDisplay() {
        if (!this.timeRangeStart || !this.timeRangeEnd) return;
        
        const duration = this.video.duration;
        const startPercent = (this.timeRangeStart / duration) * 100;
        const endPercent = (this.timeRangeEnd / duration) * 100;
        
        // Remove existing range display
        const existingRange = this.timeline.querySelector('.time-range-highlight');
        if (existingRange) {
            existingRange.remove();
        }
        
        // Add range highlight
        const rangeElement = document.createElement('div');
        rangeElement.className = 'time-range-highlight';
        rangeElement.style.position = 'absolute';
        rangeElement.style.left = `${startPercent}%`;
        rangeElement.style.width = `${endPercent - startPercent}%`;
        rangeElement.style.height = '100%';
        rangeElement.style.backgroundColor = 'rgba(59, 130, 246, 0.3)';
        rangeElement.style.border = '2px solid #3b82f6';
        rangeElement.style.pointerEvents = 'none';
        rangeElement.style.zIndex = '10';
        
        this.timeline.appendChild(rangeElement);
    }
    
    async saveTimeRangeAnnotation() {
        if (!this.currentRect || !this.currentVideoId) {
            this.showError('Please draw a bounding box first.');
            return;
        }
        
        // Validate time range
        const validation = this.validateTimeRange(this.timeRangeStart, this.timeRangeEnd);
        if (!validation.valid) {
            this.showError(validation.error);
            return;
        }
        
        // Check for overlaps (warn but don't prevent)
        if (this.checkTimeRangeOverlap(this.timeRangeStart, this.timeRangeEnd)) {
            if (!confirm('This time range overlaps with existing annotations. Continue anyway?')) {
                return;
            }
        }
        
        this.showLoading('Saving time range annotation...');
        
        try {
            const sampleInterval = parseFloat(this.sampleInterval.value);
            
            const annotationData = {
                video_id: this.currentVideoId,
                start_time: this.timeRangeStart,
                end_time: this.timeRangeEnd,
                bounding_box: this.currentRect,
                area_type: this.areaTypeSelect.value,
                confidence: 1.0,
                sample_interval: sampleInterval
            };
            
            const response = await fetch('/api/time-range-annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(annotationData)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save time range annotation');
            }
            
            const result = await response.json();
            
            this.clearCanvas();
            this.resetTimeRange();
            await this.loadAnnotations(); // Refresh regular annotations
            await this.loadTimeRangeAnnotations(); // Load time range annotations
            
            this.showSuccess(`Time range annotation saved! Created ${result.sample_frame_count} sample points over ${result.duration.toFixed(1)}s.`);
            
        } catch (error) {
            this.showError(`Failed to save time range annotation: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async loadTimeRangeAnnotations() {
        if (!this.currentVideoId) return;
        
        try {
            const response = await fetch(`/api/time-range-annotations/video/${this.currentVideoId}`);
            if (!response.ok) {
                throw new Error('Failed to load time range annotations');
            }
            
            this.timeRangeAnnotations = await response.json();
            this.updateTimeRangeDisplay();
            
        } catch (error) {
            console.error('Failed to load time range annotations:', error);
        }
    }
    
    updateTimeRangeDisplay() {
        // Add time range spans to timeline
        const existingRanges = this.timeline.querySelectorAll('.time-range-span');
        existingRanges.forEach(span => span.remove());
        
        this.timeRangeAnnotations.forEach((range, index) => {
            const duration = this.video.duration;
            const startPercent = (range.start_time / duration) * 100;
            const endPercent = (range.end_time / duration) * 100;
            
            const spanElement = document.createElement('div');
            spanElement.className = 'time-range-span';
            spanElement.style.position = 'absolute';
            spanElement.style.left = `${startPercent}%`;
            spanElement.style.width = `${endPercent - startPercent}%`;
            spanElement.style.height = '4px';
            spanElement.style.bottom = '20px';
            spanElement.style.backgroundColor = this.getColorForAreaType(range.area_type);
            spanElement.style.pointerEvents = 'none';
            spanElement.style.zIndex = '5';
            spanElement.title = `${range.area_type}: ${range.start_time.toFixed(1)}s - ${range.end_time.toFixed(1)}s`;
            
            this.timeline.appendChild(spanElement);
        });
    }
    
    // Time range validation and utility methods
    validateTimeRange(startTime, endTime) {
        if (startTime === null || endTime === null) {
            return { valid: false, error: 'Please set both start and end times' };
        }
        
        if (startTime >= endTime) {
            return { valid: false, error: 'Start time must be before end time' };
        }
        
        if (startTime < 0 || endTime > this.video.duration) {
            return { valid: false, error: 'Times must be within video duration' };
        }
        
        const minDuration = 0.1; // Minimum 0.1 seconds
        if (endTime - startTime < minDuration) {
            return { valid: false, error: `Time range must be at least ${minDuration} seconds` };
        }
        
        const maxDuration = 300; // Maximum 5 minutes
        if (endTime - startTime > maxDuration) {
            return { valid: false, error: `Time range cannot exceed ${maxDuration} seconds` };
        }
        
        return { valid: true };
    }
    
    checkTimeRangeOverlap(startTime, endTime) {
        return this.timeRangeAnnotations.some(existing => {
            return !(endTime <= existing.start_time || startTime >= existing.end_time);
        });
    }
    
    formatTimeRange(startTime, endTime) {
        const start = this.formatTime(startTime);
        const end = this.formatTime(endTime);
        const duration = (endTime - startTime).toFixed(1);
        return `${start} - ${end} (${duration}s)`;
    }
    
    getColorForAreaType(areaType) {
        const colors = {
            'chatgpt': '#10b981',
            'atuin': '#f59e0b',
            'terminal': '#8b5cf6',
            'sensitive_text': '#ef4444',
            'custom': '#6b7280'
        };
        return colors[areaType] || colors['custom'];
    }
}

// Initialize the application
const app = new VideoAnnotationApp();

// Make app globally available for event handlers
window.app = app;