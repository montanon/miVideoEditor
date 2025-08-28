"""Configuration models for ML pipeline components."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Dataset-related configuration."""

    images_dir: Path = Field(..., description="Directory with training images")
    annotations_path: Path = Field(
        ..., description="COCO-like JSON with bounding boxes"
    )
    train_split: float = Field(0.8, ge=0.1, le=0.95)
    batch_size: int = Field(4, ge=1)
    num_workers: int = Field(2, ge=0)
    image_size: int = Field(640, ge=64, description="Max side size for resizing")
    augment: bool = Field(True, description="Enable light data augmentation")

    @field_validator("images_dir", "annotations_path")
    @classmethod
    def _to_path(cls, v: Path) -> Path:
        return Path(v)


class ModelConfig(BaseModel):
    """Model-related configuration."""

    name: Literal[
        "fasterrcnn_resnet50_fpn",
        "ssd300_vgg16",
    ] = "fasterrcnn_resnet50_fpn"
    num_classes: int = Field(5, ge=2)
    pretrained: bool = True
    freeze_backbone: bool = False


class TrainConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = Field(10, ge=1)
    lr: float = Field(5e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    clip_grad_norm: float | None = Field(1.0, ge=0)
    mixed_precision: bool = True
    checkpoint_dir: Path = Field(Path("artifacts/checkpoints"))
    save_best_only: bool = True

    @field_validator("checkpoint_dir")
    @classmethod
    def _to_path(cls, v: Path) -> Path:
        return Path(v)


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    iou_threshold: float = Field(0.5, ge=0.1, le=0.9)
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(100, ge=1)


class PredictConfig(BaseModel):
    """Prediction configuration."""

    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(50, ge=1)
    image_size: int = Field(800, ge=64)


class PipelineConfig(BaseModel):
    """Full pipeline configuration bundle."""

    data: DataConfig
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    predict: PredictConfig = PredictConfig()
