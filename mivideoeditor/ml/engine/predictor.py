"""Decoupled prediction workflow for detection models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2 as T
from transformers import (
    AutoImageProcessor,
    AutoModelForInstanceSegmentation,
    AutoModelForObjectDetection,
)

from mivideoeditor.ml.config import (
    HFTrainingConfig,
    HuggingFaceModelConfig,
    HuggingFacePredictConfig,
    TorchModelConfig,
    TorchPredictConfig,
)
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
        model_cfg: TorchModelConfig,
        predict_cfg: TorchPredictConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.predict_cfg = predict_cfg or TorchPredictConfig()
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


@dataclass
class HFPrediction:
    """Hugging Face prediction."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class HFPredictor:
    """HF predictor using AutoModelForObjectDetection/InstanceSegmentation."""

    def __init__(
        self,
        model_cfg: HuggingFaceModelConfig,
        predict_cfg: HuggingFacePredictConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.predict_cfg = predict_cfg or HuggingFacePredictConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load the model."""
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_cfg.model_id,
            revision=self.model_cfg.revision,
            cache_dir=self.model_cfg.cache_dir,
        )
        if self.model_cfg.task == "detection":
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.revision,
                cache_dir=self.model_cfg.cache_dir,
            ).to(self.device)
        elif self.model_cfg.task == "instance_segmentation":
            self.model = AutoModelForInstanceSegmentation.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.revision,
                cache_dir=self.model_cfg.cache_dir,
            ).to(self.device)
        else:
            msg = "semantic_segmentation inference not integrated here"
            raise NotImplementedError(msg)
        self.model.eval()

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> HFPrediction:
        """Predict on a PIL image."""
        if self.model is None or self.processor is None:
            self.load_model()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)

        # Post-process for detection/instance segmentation
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.predict_cfg.score_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"].detach().cpu().numpy()

        if boxes.shape[0] > self.predict_cfg.max_detections:
            idx = np.argsort(-scores)[: self.predict_cfg.max_detections]
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
        return HFPrediction(boxes=boxes, scores=scores, labels=labels)

    def predict_numpy(self, image_bgr: np.ndarray) -> HFPrediction:
        """Predict on an OpenCV-style BGR numpy image."""
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            img_rgb = image_bgr[:, :, ::-1]
        else:
            img_rgb = np.stack([image_bgr] * 3, axis=-1)
        pil = Image.fromarray(img_rgb)
        return self.predict_pil(pil)


class TorchSegmentationPredictor:
    """Torch Mask R-CNN predictor returning boxes/scores/labels and binary masks."""

    def __init__(
        self,
        model_cfg: TorchModelConfig,
        predict_cfg: TorchPredictConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.predict_cfg = predict_cfg or TorchPredictConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Build Mask R-CNN with correct num_classes
        weights = (
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            if model_cfg and getattr(model_cfg, "pretrained", True)
            else None
        )
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights, num_classes=model_cfg.num_classes
        ).to(self.device)
        self.model.eval()
        size = self.predict_cfg.image_size
        self.transforms = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size, max_size=size),
            ]
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load the model checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> Prediction:
        """Predict on a PIL image."""
        tensor = self.transforms(image).to(self.device)
        outputs = self.model([tensor])[0]
        scores = outputs.get("scores", torch.empty(0))
        boxes = outputs.get("boxes", torch.empty((0, 4)))
        labels = outputs.get("labels", torch.empty(0, dtype=torch.int64))
        masks = outputs.get(
            "masks", torch.empty((0, 1, tensor.shape[-2], tensor.shape[-1]))
        )

        mask = scores >= self.predict_cfg.score_threshold
        boxes = boxes[mask][: self.predict_cfg.max_detections]
        scores = scores[mask][: self.predict_cfg.max_detections]
        labels = labels[mask][: self.predict_cfg.max_detections]
        masks = masks[mask][: self.predict_cfg.max_detections]
        # Binarize masks at 0.5
        if masks.numel() > 0:
            masks = (masks.squeeze(1) > 0.5).to(torch.uint8)

        return Prediction(
            boxes=boxes.detach().cpu().numpy(),
            scores=scores.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy(),
            masks=masks.detach().cpu().numpy() if masks.numel() > 0 else None,
        )

    def predict_numpy(self, image_bgr: np.ndarray) -> Prediction:
        """Predict on an OpenCV-style BGR numpy image."""
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            img_rgb = image_bgr[:, :, ::-1]
        else:
            img_rgb = np.stack([image_bgr] * 3, axis=-1)
        pil = Image.fromarray(img_rgb)
        return self.predict_pil(pil)
