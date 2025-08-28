"""FastAPI web application for video annotation tool."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mivideoeditor.web.routers.annotations import router as annotations_router

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Video Annotation Tool",
    description="Browser-based video annotation application for sensitive area detection",
    version="1.0.0",
)

# CORS middleware for browser access - restrictive for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include routers
app.include_router(annotations_router, prefix="/api", tags=["annotations"])

# Static file serving for frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_model=None)
async def root():
    """Serve the main annotation interface."""
    static_file = static_dir / "annotation-app.html"
    if static_file.exists():
        return FileResponse(static_file, media_type="text/html")
    return {
        "message": "Video Annotation Tool API",
        "docs": "/docs",
        "frontend": "/static/annotation-app.html"
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "video-annotation-tool",
        "version": "1.0.0"
    }


def create_app() -> FastAPI:
    """Create and return the FastAPI app."""
    return app


def run_dev_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run development server."""
    logger.info("Starting annotation app server at http://%s:%d", host, port)
    uvicorn.run(
        "mivideoeditor.web.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_dev_server()
