"""CNN-based detector for deep learning-powered detection."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mivideoeditor.core.models import BoundingBox, DetectionResult
from mivideoeditor.detection.base import BaseDetector, DetectionConfig, DetectionError

logger = logging.getLogger(__name__)

# Model architecture constants
DEFAULT_INPUT_SIZE = (512, 512)
DEFAULT_BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.5
IMAGE_CHANNELS = 3


class CNNDetector(BaseDetector):
    """Deep learning detector using Convolutional Neural Networks."""

    def __init__(
        self,
        config: DetectionConfig,
        model_path: Path | None = None,
        area_type: str = "general",
        *,
        use_gpu: bool = False,
    ) -> None:
        """Initialize CNN detector."""
        super().__init__(config)

        self.area_type = area_type
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.input_size = DEFAULT_INPUT_SIZE

        # Try to load model if path provided
        if model_path and model_path.exists():
            self.load_model(model_path)

    def _initialize_model(self) -> None:
        """Initialize model - requires model_path to be set."""
        if not self.model_path:
            msg = (
                "CNN detector requires a model_path. "
                "Use external training scripts to create a model."
            )
            raise DetectionError(msg, "NO_MODEL_PATH")

        if not self.model_path.exists():
            msg = f"Model file not found: {self.model_path}"
            raise DetectionError(msg, "MODEL_FILE_NOT_FOUND")

        try:
            self.load_model(self.model_path)
        except Exception as e:
            msg = f"Failed to load model from {self.model_path}: {e}"
            raise DetectionError(msg, "MODEL_LOAD_FAILED") from e

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect sensitive regions using CNN."""
        start_time = time.perf_counter()

        try:
            self._validate_frame(frame)

            if self.model is None:
                logger.warning("CNN model not initialized")
                return DetectionResult(
                    detections=[],
                    detection_time=time.perf_counter() - start_time,
                    detector_type="cnn",
                    timestamp=timestamp,
                    frame_metadata={"error": "model_not_initialized"},
                )

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)

            # Generate region proposals (sliding window or selective search)
            proposals = self._generate_proposals(frame)

            # Batch process proposals through CNN
            detections = self._classify_proposals(processed_frame, proposals)

            # Non-maximum suppression
            filtered_detections = self._apply_nms(detections)

            # Create result
            detection_time = time.perf_counter() - start_time
            detections_list = [
                (bbox, confidence, self.area_type)
                for bbox, confidence in filtered_detections
            ]
            result = DetectionResult(
                detections=detections_list,
                detection_time=detection_time,
                detector_type=f"cnn_{self.area_type}",
                timestamp=timestamp,
                frame_metadata={
                    "proposals_generated": len(proposals),
                    "detections_before_nms": len(detections),
                    "final_detections": len(filtered_detections),
                    "used_gpu": self.use_gpu,
                },
            )

            self._update_stats(result)

        except Exception as e:
            logger.exception("CNN detection failed")
            return DetectionResult(
                detections=[],
                detection_time=time.time() - start_time,
                detector_type=f"cnn_{self.area_type}",
                timestamp=timestamp,
                frame_metadata={"error": str(e)},
            )
        return result

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for CNN input."""
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        # Convert BGR to RGB if needed
        if (
            len(normalized.shape) == IMAGE_CHANNELS
            and normalized.shape[2] == IMAGE_CHANNELS
        ):
            normalized = normalized[:, :, ::-1]

        return normalized

    def _generate_proposals(self, frame: np.ndarray) -> list[BoundingBox]:
        """Generate region proposals for classification."""
        proposals = []
        h, w = frame.shape[:2]

        # Simple sliding window approach
        # In production, use Selective Search or RPN
        window_sizes = [(100, 100), (150, 150), (200, 200)]
        stride = 50

        for win_h, win_w in window_sizes:
            for y in range(0, h - win_h, stride):
                for x in range(0, w - win_w, stride):
                    proposals.extend([BoundingBox(x, y, win_w, win_h)])

        return proposals[:100]  # Limit proposals for performance

    def _classify_proposals(
        self, frame: np.ndarray, proposals: list[BoundingBox]
    ) -> list[tuple[BoundingBox, float]]:
        """Classify each proposal using CNN."""
        if self.model is None:
            # Fallback: random classification for testing

            return [
                (bbox, 0.75 + (hash(str(bbox)) % 100) / 1000.0)
                for bbox in proposals[:5]
            ]

        try:
            detections = []
            batch_size = DEFAULT_BATCH_SIZE

            # Process in batches
            for i in range(0, len(proposals), batch_size):
                batch_proposals = proposals[i : i + batch_size]
                batch_images = []

                # Extract and preprocess each proposal
                for bbox in batch_proposals:
                    x, y = bbox.x, bbox.y
                    w, h = bbox.width, bbox.height

                    # Extract region
                    region = frame[y : y + h, x : x + w]

                    # Resize to model input
                    resized = cv2.resize(region, self.input_size)

                    # Normalize
                    normalized = resized.astype(np.float32) / 255.0
                    batch_images.append(normalized)
                # Transpose for PyTorch (N, C, H, W)
                batch_array = np.array([img.transpose(2, 0, 1) for img in batch_images])
                batch_tensor = torch.FloatTensor(batch_array)

                if self.use_gpu and torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    # Get positive class probability (assuming binary classification)
                    confidences = probs[:, 1].cpu().numpy()

                # Filter by confidence threshold
                for bbox, conf in zip(batch_proposals, confidences, strict=True):
                    if conf > self.config.confidence_threshold:
                        detections.append((bbox, float(conf)))

        except ImportError:
            # PyTorch not available, return mock results
            return [(bbox, 0.75) for bbox in proposals[:5]]
        return detections

    def _apply_nms(
        self, detections: list[tuple[BoundingBox, float]], iou_threshold: float = 0.3
    ) -> list[tuple[BoundingBox, float]]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)

        keep = []
        while sorted_detections:
            # Take highest confidence detection
            best = sorted_detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            sorted_detections = [
                det for det in sorted_detections if best[0].iou(det[0]) < iou_threshold
            ]

        return keep[: self.config.max_regions_per_frame]

    def load_model(self, path: Path) -> None:
        """Load trained CNN model from PyTorch file."""
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise DetectionError(msg, "MODEL_FILE_NOT_FOUND")

        try:
            # Load the checkpoint
            checkpoint = torch.load(path, map_location="cpu")

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                # Complete model saved in dictionary
                self.model = checkpoint["model"]
                # Load metadata if available
                self.area_type = checkpoint.get("area_type", self.area_type)
                self.input_size = checkpoint.get("input_size", DEFAULT_INPUT_SIZE)
            elif callable(checkpoint):
                # Complete model saved directly
                self.model = checkpoint
            else:
                msg = (
                    "Invalid model format. Expected complete model or dict with 'model'"
                    "  key. State dicts not supported - save complete model instead."
                )
                raise DetectionError(msg, "INVALID_MODEL_FORMAT")

            # Move to appropriate device
            if self.use_gpu:
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                elif torch.backends.mps.is_available():
                    self.model = self.model.to("mps")

            self.model.eval()
            self.is_trained = True
            logger.info("CNN model loaded from %s", path)

        except ImportError:
            msg = "PyTorch not installed. Install with: uv add torch"
            raise DetectionError(msg, "MISSING_PYTORCH")

        except Exception as e:
            msg = f"Failed to load CNN model: {e}"
            raise DetectionError(msg, "MODEL_LOAD_FAILED") from e


class YOLODetector(BaseDetector):
    """YOLO-based detector for real-time object detection."""

    def __init__(
        self,
        config: DetectionConfig,
        model_version: str = "yolov5s",
        area_type: str = "general",
    ) -> None:
        """Initialize YOLO detector."""
        super().__init__(config)

        self.model_version = model_version
        self.area_type = area_type
        self.model = None

        self._initialize_yolo()

    def _initialize_yolo(self) -> None:
        """Initialize YOLO model."""
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load(
                "ultralytics/yolov5", self.model_version, pretrained=True
            )

            # Configure model
            self.model.conf = self.config.confidence_threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.max_det = self.config.max_regions_per_frame

            logger.info("YOLO %s model initialized", self.model_version)

        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("Failed to initialize YOLO: %s", e)
            self.model = None

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect using YOLO model."""
        start_time = time.perf_counter()

        try:
            self._validate_frame(frame)

            if self.model is None:
                return DetectionResult(
                    detections=[],
                    detection_time=time.perf_counter() - start_time,
                    detector_type=f"yolo_{self.model_version}",
                    timestamp=timestamp,
                    frame_metadata={"error": "model_not_initialized"},
                )

            # Run YOLO inference
            results = self.model(frame)

            # Parse detections
            detections = []
            for *box, conf, _ in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2 = map(int, box)
                bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
                detections.append((bbox, float(conf)))

            # Filter by area
            filtered = [
                (bbox, conf)
                for bbox, conf in detections
                if bbox.area >= self.config.min_detection_area
            ]

            detection_time = time.perf_counter() - start_time

            detections_list = [
                (bbox, conf, "general")  # YOLO detects general objects
                for bbox, conf in filtered
            ]
            return DetectionResult(
                detections=detections_list,
                detection_time=detection_time,
                detector_type=f"yolo_{self.model_version}",
                timestamp=timestamp,
                frame_metadata={
                    "raw_detections": len(detections),
                    "filtered_detections": len(filtered),
                },
            )

        except Exception as e:
            logger.exception("YOLO detection failed")
            return DetectionResult(
                detections=[],
                detection_time=time.perf_counter() - start_time,
                detector_type=f"yolo_{self.model_version}",
                timestamp=timestamp,
                frame_metadata={"error": str(e)},
            )
