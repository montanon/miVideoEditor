"""Unified detector adapter that can run Torch or HF predictors behind one API."""

from __future__ import annotations

import logging

import numpy as np

from mivideoeditor.core.models import BoundingBox, DetectionResult
from mivideoeditor.detection.base import BaseDetector, DetectionConfig
from mivideoeditor.ml.backends.base import BasePredictor
from mivideoeditor.ml.backends.factory import get_predictor

logger = logging.getLogger(__name__)


class Detector(BaseDetector):
    """Single detector that delegates to a configured backend predictor.backends factory. Supports Torch and HF without exposing separate classes."""

    def __init__(
        self,
        config: DetectionConfig,
        *,
        label_map: dict[int, str],
        backend: str = "torch",
        predictor: BasePredictor | None = None,
        **predictor_kwargs: object,
    ) -> None:
        super().__init__(config)
        self.label_map = label_map
        self.detector_type = f"detector_{backend}"
        self.predictor: BasePredictor = predictor or get_predictor(
            backend, task="detection", **predictor_kwargs
        )
        self.is_trained = True

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect sensitive regions in a single frame."""
        self._validate_frame(frame)
        pred = self.predictor.predict_numpy(frame)

        detections: list[tuple[BoundingBox, float, str]] = []
        for (x1, y1, x2, y2), score, label in zip(
            pred.boxes, pred.scores, pred.labels, strict=True
        ):
            # Enforce area constraints from config
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            area = w * h
            if area < self.config.min_detection_area:
                continue
            if self.config.max_detection_area and area > self.config.max_detection_area:
                continue

            area_type = self.label_map.get(int(label), "custom")
            bbox = BoundingBox(x=int(x1), y=int(y1), width=w, height=h)
            detections.append((bbox, float(score), area_type))

        result = DetectionResult(
            detections=detections[: self.config.max_regions_per_frame],
            detection_time=0.0,  # populated by caller if needed
            detector_type=self.detector_type,
            timestamp=timestamp,
            frame_metadata={"count": len(detections)},
        )
        self._update_stats(result)
        return result
