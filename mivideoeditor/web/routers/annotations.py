"""Annotation API router integrating with existing storage and detection services."""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from mivideoeditor.core.models import BoundingBox, SensitiveArea, TimeRangeAnnotation, VideoRecord
from mivideoeditor.storage.annotation_service import AnnotationService
from mivideoeditor.storage.file_manager import FileManager, FileManagerConfig
from mivideoeditor.storage.service import StorageService, StorageConfig
from mivideoeditor.utils.video import VideoUtils

logger = logging.getLogger(__name__)

# Security constants
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB for long videos
MAX_FILENAME_LENGTH = 100
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for large file handling

# Initialize services
storage_config = StorageConfig()
storage_service = StorageService(storage_config)
storage_service.initialize()

file_manager_config = FileManagerConfig()
file_manager = FileManager(Path("data"), file_manager_config)

annotation_service = AnnotationService(storage_service, file_manager)
video_utils = VideoUtils()

router = APIRouter()

# Request/Response Models
class AnnotationRequest(BaseModel):
    """Request model for creating annotations."""
    video_id: str = Field(..., description="Video identifier")
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds")
    bounding_box: Dict[str, int] = Field(..., description="Bounding box coordinates")
    area_type: str = Field(..., description="Type of sensitive area")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class AnnotationResponse(BaseModel):
    """Response model for annotation data."""
    id: str
    video_id: str
    timestamp: float
    bounding_box: Dict[str, int]
    area_type: str
    confidence: float
    frame_number: Optional[int] = None
    image_path: Optional[str] = None
    created_at: str


class VideoUploadResponse(BaseModel):
    """Response model for video upload."""
    video_id: str
    filename: str
    duration: float
    width: int
    height: int
    frame_rate: float
    file_size: int


class VideoFrameRequest(BaseModel):
    """Request model for frame extraction."""
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds")


class TimeRangeAnnotationRequest(BaseModel):
    """Request model for creating time range annotations."""
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    bounding_box: Dict[str, int] = Field(..., description="Bounding box coordinates")
    area_type: str = Field(..., description="Type of sensitive area")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    sample_interval: float = Field(default=1.0, ge=0.1, le=10.0, description="Frame sampling interval in seconds")


class TimeRangeAnnotationResponse(BaseModel):
    """Response model for time range annotation data."""
    id: str
    video_id: str
    start_time: float
    end_time: float
    duration: float
    bounding_box: Dict[str, int]
    area_type: str
    confidence: float
    sample_frame_count: int
    created_at: str


# Video Management Endpoints

@router.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and extract metadata."""
    try:
        # Validate file type - check both content type and file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        
        is_valid_type = (
            (file.content_type and file.content_type.startswith('video/')) or
            file_ext in valid_extensions
        )
        
        if not is_valid_type:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            )
        
        # Validate file size
        if file.size and file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024*1024)}GB"
            )
        
        # Sanitize filename
        if not file.filename or len(file.filename) > MAX_FILENAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename length"
            )
        
        # Remove dangerous characters and path components
        safe_filename = Path(file.filename).name  # Remove any path components
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', safe_filename)
        
        if not safe_filename or safe_filename in ['.', '..']:
            safe_filename = f"video_{int(time.time())}{file_ext}"
        
        # Create unique video ID with sanitized filename
        video_id = f"vid_{int(time.time())}_{safe_filename}"
        
        # Prepare file path
        original_ext = file_ext or '.mp4'
        video_path = file_manager.directories["videos"] / f"{video_id}{original_ext}"
        
        # Ensure the directory exists and is safe
        video_dir = video_path.parent
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Additional path safety check
        if not str(video_path).startswith(str(video_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Stream write large video file in chunks
        bytes_written = 0
        try:
            with open(video_path, "wb") as f:
                while chunk := await file.read(CHUNK_SIZE):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_SIZE:
                        # Clean up partial file
                        f.close()
                        video_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail="File too large during upload"
                        )
                    f.write(chunk)
                    
            if bytes_written == 0:
                video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Empty file uploaded")
                
        except OSError as e:
            video_path.unlink(missing_ok=True)
            logger.error(f"Failed to write video file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save video file")
        
        # Extract video metadata
        try:
            video_info = video_utils.get_video_info(video_path)
        except Exception as e:
            # Clean up file on error
            if video_path.exists():
                video_path.unlink()
            logger.error(f"Failed to extract video info: {e}")
            raise HTTPException(status_code=400, detail="Invalid video file or unsupported format.")
        
        # Store video record in database
        from mivideoeditor.storage.models import VideoRecord
        video_record = VideoRecord(
            id=video_id,
            filename=safe_filename,
            filepath=video_path,
            duration=video_info.duration,
            frame_rate=video_info.frame_rate,
            width=video_info.width,
            height=video_info.height,
            file_size=bytes_written,
            codec=video_info.codec,
            metadata={"original_filename": file.filename}
        )
        
        storage_service.save_video(video_record)
        
        logger.info(f"Video uploaded successfully: {video_id}")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=safe_filename,
            duration=video_info.duration,
            width=video_info.width,
            height=video_info.height,
            frame_rate=video_info.frame_rate,
            file_size=bytes_written
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Video upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """Get video metadata."""
    try:
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "video_id": video_record.id,
            "filename": video_record.filename,
            "duration": video_record.duration,
            "width": video_record.width,
            "height": video_record.height,
            "frame_rate": video_record.frame_rate,
            "file_size": video_record.file_size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get video info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video information")


@router.get("/videos/{video_id}/frame")
async def get_video_frame(video_id: str, timestamp: float):
    """Extract and return a frame at the specified timestamp."""
    try:
        # Get video record
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Validate timestamp with tolerance for floating point precision
        if timestamp < 0 or timestamp > (video_record.duration + 0.1):
            raise HTTPException(
                status_code=400, 
                detail=f"Timestamp {timestamp} outside video duration (0-{video_record.duration})"
            )
        
        # Extract frame
        video_path = video_record.filepath
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")
        
        # Use cv2 to extract frame with proper resource management
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video file")
            
            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                raise HTTPException(status_code=400, detail="Failed to extract frame at specified timestamp")
            
            # Encode frame as JPEG with error handling
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not success or buffer is None:
                raise HTTPException(status_code=500, detail="Failed to encode frame")
            
            # Return as streaming response
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg",
                headers={"Content-Disposition": f"inline; filename=frame_{video_id}_{timestamp:.2f}.jpg"}
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during frame extraction: {e}")
            raise HTTPException(status_code=500, detail="Frame extraction failed")
        finally:
            if cap is not None:
                cap.release()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Frame extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Frame extraction failed")


# Annotation CRUD Endpoints

@router.post("/annotations", response_model=AnnotationResponse)
async def create_annotation(request: AnnotationRequest):
    """Create a new annotation with frame extraction."""
    try:
        # Get video record
        video_record = storage_service.get_video(request.video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Validate bounding box
        try:
            # Extract and validate bounding box values
            x = request.bounding_box.get("x", 0)
            y = request.bounding_box.get("y", 0)
            width = request.bounding_box.get("width", 0)
            height = request.bounding_box.get("height", 0)
            
            # Validate bounding box constraints
            if not all(isinstance(val, (int, float)) and val >= 0 for val in [x, y, width, height]):
                raise ValueError("Bounding box coordinates must be non-negative numbers")
            
            if width < 5 or height < 5:
                raise ValueError("Bounding box too small (minimum 5x5 pixels)")
                
            if width > video_record.width or height > video_record.height:
                raise ValueError("Bounding box exceeds video dimensions")
                
            if x + width > video_record.width or y + height > video_record.height:
                raise ValueError("Bounding box extends beyond video boundaries")
            
            bbox = BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
            
        except (KeyError, ValueError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid bounding box: {e}")
        
        # Extract frame at timestamp with proper resource management
        video_path = video_record.filepath
        cap = None
        frame = None
        frame_number = 0
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for frame extraction")
                
            cap.set(cv2.CAP_PROP_POS_MSEC, request.timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                raise HTTPException(status_code=400, detail="Failed to extract frame for annotation")
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Frame extraction failed during annotation: {e}")
            raise HTTPException(status_code=500, detail="Frame extraction failed")
        finally:
            if cap is not None:
                cap.release()
        
        # Create SensitiveArea object
        sensitive_area = SensitiveArea(
            timestamp=request.timestamp,
            bounding_box=bbox,
            area_type=request.area_type,
            confidence=request.confidence,
            metadata={
                "video_id": request.video_id,
                "frame_number": frame_number,
                "created_by": "manual_annotation",
                "creation_method": "browser_app"
            }
        )
        
        # Save annotation with frame image
        annotation_id = annotation_service.save_annotation(sensitive_area, frame)
        
        # Get saved annotation for response
        saved_annotation = annotation_service.load_annotation(annotation_id)
        if not saved_annotation:
            raise HTTPException(status_code=500, detail="Failed to retrieve saved annotation")
        
        logger.info(f"Annotation created: {annotation_id} at {request.timestamp}s")
        
        return AnnotationResponse(
            id=saved_annotation.id,
            video_id=request.video_id,
            timestamp=saved_annotation.timestamp,
            bounding_box={
                "x": saved_annotation.bounding_box.x,
                "y": saved_annotation.bounding_box.y,
                "width": saved_annotation.bounding_box.width,
                "height": saved_annotation.bounding_box.height,
            },
            area_type=saved_annotation.area_type,
            confidence=saved_annotation.confidence,
            frame_number=frame_number,
            image_path=str(saved_annotation.image_path) if saved_annotation.image_path else None,
            created_at=saved_annotation.metadata.get("created_at", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Annotation creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Annotation creation failed: {str(e)}")


@router.get("/annotations/video/{video_id}", response_model=List[AnnotationResponse])
async def get_video_annotations(video_id: str):
    """Get all annotations for a video."""
    try:
        # Verify video exists
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get annotations
        annotations = annotation_service.get_annotations_for_video(video_id)
        
        # Convert to response format
        response_annotations = []
        for annotation in annotations:
            response_annotations.append(AnnotationResponse(
                id=annotation.id,
                video_id=video_id,
                timestamp=annotation.timestamp,
                bounding_box={
                    "x": annotation.bounding_box.x,
                    "y": annotation.bounding_box.y,
                    "width": annotation.bounding_box.width,
                    "height": annotation.bounding_box.height,
                },
                area_type=annotation.area_type,
                confidence=annotation.confidence,
                frame_number=annotation.metadata.get("frame_number"),
                image_path=str(annotation.image_path) if annotation.image_path else None,
                created_at=annotation.metadata.get("created_at", "")
            ))
        
        return response_annotations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get annotations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve annotations")


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation."""
    try:
        success = annotation_service.delete_annotation(annotation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        logger.info(f"Annotation deleted: {annotation_id}")
        return {"message": "Annotation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete annotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete annotation")


# Statistics and Export Endpoints

@router.get("/annotations/video/{video_id}/stats")
async def get_annotation_stats(video_id: str):
    """Get annotation statistics for a video."""
    try:
        # Verify video exists
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get statistics
        stats = annotation_service.get_annotation_statistics(video_id)
        
        return {
            "video_id": video_id,
            "video_duration": video_record.duration,
            **stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get annotation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/annotations/video/{video_id}/export")
async def export_annotations(video_id: str, format: str = "json"):
    """Export annotations in specified format."""
    try:
        # Verify video exists
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if format not in ["json", "coco"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'coco'")
        
        # Export annotations
        export_path = annotation_service.export_annotations(video_id, format)
        
        # Read exported file
        with open(export_path, 'r') as f:
            content = f.read()
        
        # Determine media type
        media_type = "application/json"
        filename = f"annotations_{video_id}_{format}.json"
        
        return StreamingResponse(
            io.StringIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail="Export failed")


# Time Range Annotation Endpoints

@router.post("/time-range-annotations", response_model=TimeRangeAnnotationResponse)
async def create_time_range_annotation(request: TimeRangeAnnotationRequest):
    """Create a new time range annotation with batch frame extraction."""
    try:
        # Get video record
        video_record = storage_service.get_video(request.video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Validate time range
        if request.end_time <= request.start_time:
            raise HTTPException(
                status_code=400, 
                detail="end_time must be greater than start_time"
            )
        
        # Validate time range is within video duration
        if request.start_time < 0 or request.end_time > (video_record.duration + 0.1):
            raise HTTPException(
                status_code=400,
                detail=f"Time range [{request.start_time}, {request.end_time}] outside video duration (0-{video_record.duration})"
            )
        
        # Validate bounding box (reuse existing validation logic)
        try:
            x = request.bounding_box.get("x", 0)
            y = request.bounding_box.get("y", 0)
            width = request.bounding_box.get("width", 0)
            height = request.bounding_box.get("height", 0)
            
            if not all(isinstance(val, (int, float)) and val >= 0 for val in [x, y, width, height]):
                raise ValueError("Bounding box coordinates must be non-negative numbers")
            
            if width < 5 or height < 5:
                raise ValueError("Bounding box too small (minimum 5x5 pixels)")
                
            if width > video_record.width or height > video_record.height:
                raise ValueError("Bounding box exceeds video dimensions")
                
            if x + width > video_record.width or y + height > video_record.height:
                raise ValueError("Bounding box extends beyond video boundaries")
            
            bbox = BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
            
        except (KeyError, ValueError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid bounding box: {e}")
        
        # Create TimeRangeAnnotation
        time_range = TimeRangeAnnotation(
            start_time=request.start_time,
            end_time=request.end_time,
            bounding_box=bbox,
            area_type=request.area_type,
            confidence=request.confidence,
            metadata={
                "video_id": request.video_id,
                "sample_interval": request.sample_interval,
                "created_by": "manual_annotation",
                "creation_method": "time_range_browser_app"
            }
        )
        
        # Convert to individual SensitiveArea annotations for storage
        sensitive_areas = time_range.to_sensitive_areas(request.sample_interval)
        
        # Save all individual annotations
        saved_annotations = []
        for i, area in enumerate(sensitive_areas):
            # Extract frame for this timestamp
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_record.filepath))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_MSEC, area.timestamp * 1000)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Save annotation with frame
                        annotation_id = annotation_service.save_annotation(area, frame)
                        saved_annotations.append(annotation_id)
                    else:
                        # Save annotation without frame
                        annotation_id = annotation_service.save_annotation(area)
                        saved_annotations.append(annotation_id)
            except Exception as e:
                logger.warning(f"Failed to extract frame for timestamp {area.timestamp}: {e}")
                # Save annotation without frame
                annotation_id = annotation_service.save_annotation(area)
                saved_annotations.append(annotation_id)
            finally:
                if cap is not None:
                    cap.release()
        
        logger.info(
            f"Time range annotation created: {time_range.id} "
            f"({request.start_time:.1f}s-{request.end_time:.1f}s) "
            f"with {len(saved_annotations)} sample points"
        )
        
        return TimeRangeAnnotationResponse(
            id=time_range.id,
            video_id=request.video_id,
            start_time=time_range.start_time,
            end_time=time_range.end_time,
            duration=time_range.duration,
            bounding_box={
                "x": time_range.bounding_box.x,
                "y": time_range.bounding_box.y,
                "width": time_range.bounding_box.width,
                "height": time_range.bounding_box.height,
            },
            area_type=time_range.area_type,
            confidence=time_range.confidence,
            sample_frame_count=len(sensitive_areas),
            created_at=time_range.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Time range annotation creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Time range annotation creation failed: {str(e)}")


@router.get("/time-range-annotations/video/{video_id}")
async def get_video_time_ranges(video_id: str):
    """Get all time range annotations for a video by analyzing individual annotations."""
    try:
        # Verify video exists
        video_record = storage_service.get_video(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get all annotations for the video
        annotations = annotation_service.get_annotations_for_video(video_id)
        
        # Group annotations by source_range_id if they exist
        time_ranges = {}
        
        for annotation in annotations:
            source_range_id = annotation.metadata.get("source_range_id")
            if source_range_id:
                if source_range_id not in time_ranges:
                    time_ranges[source_range_id] = {
                        "id": source_range_id,
                        "video_id": video_id,
                        "start_time": annotation.metadata.get("range_start", annotation.timestamp),
                        "end_time": annotation.metadata.get("range_end", annotation.timestamp),
                        "bounding_box": {
                            "x": annotation.bounding_box.x,
                            "y": annotation.bounding_box.y,
                            "width": annotation.bounding_box.width,
                            "height": annotation.bounding_box.height,
                        },
                        "area_type": annotation.area_type,
                        "confidence": annotation.confidence,
                        "sample_frame_count": 0,
                        "created_at": annotation.metadata.get("created_at", ""),
                    }
                
                time_ranges[source_range_id]["sample_frame_count"] += 1
        
        # Convert to list and add duration
        result = []
        for time_range in time_ranges.values():
            time_range["duration"] = time_range["end_time"] - time_range["start_time"]
            result.append(time_range)
        
        # Sort by start time
        result.sort(key=lambda x: x["start_time"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get time range annotations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve time range annotations")


@router.put("/videos/{video_id}/replace", response_model=VideoUploadResponse)
async def replace_video(video_id: str, file: UploadFile = File(...)):
    """Replace an existing video with a new one, preserving the video ID."""
    try:
        # Get existing video record
        existing_video = storage_service.get_video_by_id(video_id)
        if not existing_video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Validate file type - same as upload
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        
        is_valid_type = (
            (file.content_type and file.content_type.startswith('video/')) or
            file_ext in valid_extensions
        )
        
        if not is_valid_type:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            )
        
        # Validate file size
        if file.size and file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024*1024)}GB"
            )
        
        # Create new filename with sanitization (same as upload)
        if not file.filename or len(file.filename) > MAX_FILENAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename length"
            )
        
        # Remove dangerous characters and path components
        safe_filename = Path(file.filename).name  # Remove any path components
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', safe_filename)
        
        if not safe_filename or safe_filename in ['.', '..']:
            safe_filename = f"video_{int(time.time())}{file_ext}"
        
        # Use existing video ID with new filename
        original_ext = file_ext or '.mp4'
        file_path = file_manager.directories["videos"] / f"{video_id}{original_ext}"
        
        # Clean up old video file if it exists
        old_video_path = Path(existing_video.filepath)
        if old_video_path.exists():
            try:
                old_video_path.unlink()
                logger.info(f"Removed old video file: {old_video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old video file {old_video_path}: {e}")
        
        # Save new video file with streaming
        total_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    # Clean up partial file
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024*1024)}GB"
                    )
                buffer.write(chunk)
        
        # Extract video metadata using VideoUtils
        try:
            metadata = VideoUtils.get_video_info(str(file_path))
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")
        
        # Update video record with new information
        video_record = VideoRecord(
            id=video_id,  # Keep same ID
            filename=safe_filename,
            filepath=str(file_path),
            filesize=total_size,
            duration=metadata.duration,
            width=metadata.width, 
            height=metadata.height,
            fps=metadata.fps,
            upload_timestamp=datetime.utcnow(),
            format=metadata.format if hasattr(metadata, 'format') else 'unknown'
        )
        
        # Save updated video record
        storage_service.save_video(video_record)
        
        logger.info(f"Video replaced successfully: {video_id} -> {file_path}")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=safe_filename,
            duration=metadata.duration,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            message="Video replaced successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Video replacement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video replacement failed: {str(e)}")


@router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its associated annotations."""
    try:
        # Get video record
        video_record = storage_service.get_video_by_id(video_id)
        if not video_record:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Delete all annotations for this video
        annotations = storage_service.get_annotations_by_video(video_id)
        for annotation in annotations:
            storage_service.delete_annotation(annotation.id)
        
        # Delete all time range annotations for this video (if they exist)
        try:
            time_range_annotations = storage_service.get_time_range_annotations_by_video(video_id)
            for tr_annotation in time_range_annotations:
                storage_service.delete_time_range_annotation(tr_annotation.id)
        except AttributeError:
            # Time range annotations might not be implemented in storage yet
            pass
        
        # Delete video file
        video_path = Path(video_record.filepath)
        if video_path.exists():
            try:
                video_path.unlink()
                logger.info(f"Deleted video file: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete video file {video_path}: {e}")
        
        # Delete video record from storage
        storage_service.delete_video(video_id)
        
        logger.info(f"Video deleted successfully: {video_id}")
        
        return {"message": "Video and all associated data deleted successfully", "video_id": video_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Video deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video deletion failed: {str(e)}")