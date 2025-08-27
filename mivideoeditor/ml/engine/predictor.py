"""Decoupled prediction workflow for detection models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from mivideoeditor.ml.config import ModelConfig, PredictConfig
from mivideoeditor.ml.models import build_model

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction result."""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)


class Predictor:
    """Predictor loads a model checkpoint and runs inference on images."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        predict_cfg: PredictConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.predict_cfg = predict_cfg or PredictConfig()
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = build_model(model_cfg).to(self.device)
        self.model.eval()
        self.transforms = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(predict_cfg.image_size, max_size=predict_cfg.image_size),
            ]
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model weights from Trainer checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> Prediction:
        """Predict on a PIL image."""
        tensor = self.transforms(image).to(self.device)
        outputs = self.model([tensor])[0]
        scores = outputs.get("scores", torch.empty(0))
        boxes = outputs.get("boxes", torch.empty((0, 4)))
        labels = outputs.get("labels", torch.empty(0, dtype=torch.int64))

        mask = scores >= self.predict_cfg.score_threshold
        boxes = boxes[mask][: self.predict_cfg.max_detections]
        scores = scores[mask][: self.predict_cfg.max_detections]
        labels = labels[mask][: self.predict_cfg.max_detections]

        return Prediction(
            boxes=boxes.detach().cpu().numpy(),
            scores=scores.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy(),
        )

    def predict_numpy(self, image_bgr: np.ndarray) -> Prediction:
        """Predict on an OpenCV-style BGR numpy image."""
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            img_rgb = image_bgr[:, :, ::-1]
        else:
            # assume grayscale
            img_rgb = np.stack([image_bgr] * 3, axis=-1)
        pil = Image.fromarray(img_rgb)
        return self.predict_pil(pil)
