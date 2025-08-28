"""Model builder utilities for detectors."""

from __future__ import annotations

import logging
from typing import Any

import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mivideoeditor.ml.config import TorchModelConfig

logger = logging.getLogger(__name__)


def build_model(cfg: TorchModelConfig) -> nn.Module:
    """Create a torchvision detection model according to config."""
    num_classes = cfg.num_classes
    name = cfg.name

    if name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=(
                torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                if cfg.pretrained
                else None
            ),
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif name == "ssd300_vgg16":
        model = torchvision.models.detection.ssd300_vgg16(
            weights=(
                torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
                if cfg.pretrained
                else None
            ),
            num_classes=num_classes,
        )
    else:
        msg = f"Unsupported model name: {name}"
        raise ValueError(msg)

    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen for fine-tuning")
    return model


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Count the number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
