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


class HuggingFaceModelConfig(BaseModel):
    """Config for Hugging Face detection/segmentation models."""

    model_id: str = Field(
        "facebook/detr-resnet-50",
        description="HF Hub model id or local path",
    )
    revision: str | None = Field(None, description="Optional revision/commit")
    num_labels: int | None = Field(
        None, description="Override number of labels when training"
    )
    cache_dir: Path | None = Field(None, description="Optional HF cache dir")
    task: Literal[
        "detection",
        "instance_segmentation",
        "semantic_segmentation",
    ] = Field("detection", description="Task type")

    @field_validator("cache_dir")
    @classmethod
    def _to_path_hf(cls, v: Path | None) -> Path | None:
        return None if v is None else Path(v)


class HuggingFacePredictConfig(BaseModel):
    """Config for Hugging Face prediction."""

    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(50, ge=1)


class HFTrainingConfig(BaseModel):
    """Transformers Trainer configuration subset."""

    output_dir: Path = Field(Path("artifacts/hf"))
    num_train_epochs: int = Field(10, ge=1)
    per_device_train_batch_size: int = Field(2, ge=1)
    per_device_eval_batch_size: int = Field(2, ge=1)
    learning_rate: float = Field(5e-5, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    lr_scheduler_type: str = Field("cosine")
    warmup_ratio: float = Field(0.0, ge=0.0, le=0.3)
    fp16: bool = Field(True)
    logging_steps: int = Field(50, ge=1)
    evaluation_strategy: Literal["no", "epoch", "steps"] = Field("epoch")
    save_strategy: Literal["epoch", "steps"] = Field("epoch")
    gradient_accumulation_steps: int = Field(1, ge=1)
    seed: int = Field(42)

    @field_validator("output_dir")
    @classmethod
    def _to_path2(cls, v: Path) -> Path:
        return Path(v)
