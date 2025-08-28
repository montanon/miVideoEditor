"""Startup script for the Video Annotation Tool."""

import logging
import sys

from mivideoeditor.web.app import run_dev_server


def main() -> None:
    """Run Main file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("🎬 Video Annotation Tool")
    logger.info("=" * 50)
    logger.info("Starting annotation application server...")
    logger.info("✅ Backend API: FastAPI with annotation endpoints")
    logger.info("✅ Frontend: Browser-based video annotation interface")
    logger.info("✅ Storage: SQLite database with file management")
    logger.info("✅ Integration: Direct compatibility with detection training")
    logger.info("")
    logger.info("📖 Usage:")
    logger.info("  1. Upload a screen recording video")
    logger.info("  2. Draw bounding boxes around sensitive areas")
    logger.info("  3. Save annotations with area type labels")
    logger.info("  4. Export for training detection models")
    logger.info("")
    logger.info("🌐 Access the app at: http://127.0.0.1:8000")
    logger.info("📚 API docs at: http://127.0.0.1:8000/docs")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 50)

    try:
        run_dev_server(host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        logger.info("\n👋 Annotation app server stopped")
    except (OSError, RuntimeError, ImportError) as e:
        msg = f"\n❌ Error: {e}"
        logger.exception(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
