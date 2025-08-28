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

        // Timeline
        this.timeline.addEventListener('click', (e) => this.onTimelineClick(e));

        // Export buttons
        this.exportJsonBtn.addEventListener('click', () => this.exportAnnotations('json'));
        this.exportCocoBtn.addEventListener('click', () => this.exportAnnotations('coco'));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Window resize
        window.addEventListener('resize', () => this.resizeCanvas());
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
        this.canvas.width = this.video.videoWidth || rect.width;
        this.canvas.height = this.video.videoHeight || rect.height;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        this.canvas.classList.add('drawing');
    }

    // Canvas Drawing Methods
    onCanvasMouseDown(event) {
        if (this.video.paused) {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;

            this.startX = (event.clientX - rect.left) * scaleX;
            this.startY = (event.clientY - rect.top) * scaleY;
            this.isDrawing = true;

            this.clearCanvas();
        }
    }

    onCanvasMouseMove(event) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const currentX = (event.clientX - rect.left) * scaleX;
        const currentY = (event.clientY - rect.top) * scaleY;

        this.drawBoundingBox(this.startX, this.startY, currentX, currentY);
    }

    onCanvasMouseUp(event) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const endX = (event.clientX - rect.left) * scaleX;
        const endY = (event.clientY - rect.top) * scaleY;

        this.isDrawing = false;

        // Calculate bounding box
        const x = Math.min(this.startX, endX);
        const y = Math.min(this.startY, endY);
        const width = Math.abs(endX - this.startX);
        const height = Math.abs(endY - this.startY);

        if (width > 10 && height > 10) {
            this.currentRect = { x: Math.round(x), y: Math.round(y), width: Math.round(width), height: Math.round(height) };
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
        this.video.currentTime = position * this.video.duration;
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
}

// Initialize the application
const app = new VideoAnnotationApp();

// Make app globally available for event handlers
window.app = app;