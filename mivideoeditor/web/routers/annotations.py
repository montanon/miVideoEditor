"""Annotation API router integrating with existing storage and detection services."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from mivideoeditor.core.models import BoundingBox, SensitiveArea
from mivideoeditor.storage.annotation_service import AnnotationService
from mivideoeditor.storage.file_manager import FileManager, FileManagerConfig
from mivideoeditor.storage.service import StorageService, StorageConfig
from mivideoeditor.utils.video import VideoUtils

logger = logging.getLogger(__name__)

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
        
        # Create unique video ID
        video_id = f"vid_{int(time.time())}_{file.filename}"
        
        # Save uploaded file with original extension
        video_content = await file.read()
        original_ext = file_ext or '.mp4'
        video_path = file_manager.directories["videos"] / f"{video_id}{original_ext}"
        
        # Write video file
        with open(video_path, "wb") as f:
            f.write(video_content)
        
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
            filename=file.filename,
            filepath=video_path,
            duration=video_info.duration,
            frame_rate=video_info.frame_rate,
            width=video_info.width,
            height=video_info.height,
            file_size=len(video_content),
            codec=video_info.codec,
            metadata={"original_filename": file.filename}
        )
        
        storage_service.save_video(video_record)
        
        logger.info(f"Video uploaded successfully: {video_id}")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            duration=video_info.duration,
            width=video_info.width,
            height=video_info.height,
            frame_rate=video_info.frame_rate,
            file_size=len(video_content)
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
        
        # Validate timestamp
        if timestamp < 0 or timestamp > video_record.duration:
            raise HTTPException(
                status_code=400, 
                detail=f"Timestamp {timestamp} outside video duration (0-{video_record.duration})"
            )
        
        # Extract frame
        video_path = video_record.filepath
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")
        
        # Use cv2 to extract frame
        cap = cv2.VideoCapture(str(video_path))
        try:
            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                raise HTTPException(status_code=400, detail="Failed to extract frame at specified timestamp")
            
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame")
            
            # Return as streaming response
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg",
                headers={"Content-Disposition": f"inline; filename=frame_{video_id}_{timestamp:.2f}.jpg"}
            )
            
        finally:
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
            bbox = BoundingBox(
                x=request.bounding_box["x"],
                y=request.bounding_box["y"],
                width=request.bounding_box["width"],
                height=request.bounding_box["height"]
            )
        except (KeyError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid bounding box: {e}")
        
        # Extract frame at timestamp
        video_path = video_record.filepath
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, request.timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                raise HTTPException(status_code=400, detail="Failed to extract frame for annotation")
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
        finally:
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