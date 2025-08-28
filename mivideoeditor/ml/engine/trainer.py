"""Decoupled training workflow for detection models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForInstanceSegmentation,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

from mivideoeditor.ml.config import (
    DataConfig,
    EvalConfig,
    HFTrainingConfig,
    HuggingFaceModelConfig,
    ModelConfig,
    TrainConfig,
)
from mivideoeditor.ml.data import DetectionDataset, collate_fn
from mivideoeditor.ml.models import build_model, count_parameters

logger = logging.getLogger(__name__)


@dataclass
class TrainArtifacts:
    """Artifacts from training."""

    checkpoint_path: Path
    best_epoch: int
    best_metric: float
    params: dict[str, Any]


class DetectionTrainer:
    """Detection trainer encapsulates dataset, model, loop and checkpointing."""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg or EvalConfig()
        # Prefer explicit device if provided; otherwise pick CUDA, then MPS, else CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Data
        self.ds_train = DetectionDataset(data_cfg, split="train")
        self.ds_val = DetectionDataset(data_cfg, split="val")
        self.train_loader = DataLoader(
            self.ds_train,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.ds_val,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            collate_fn=collate_fn,
        )

        # Model
        self.model = build_model(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.mixed_precision)

        trainable = count_parameters(self.model)
        logger.info("Model parameters: %s", trainable)

        # Checkpoint dir
        self.ckpt_dir = train_cfg.checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _step(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[float, dict[str, float]]:
        """Train one step."""
        self.model.train()
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.autocast(self.device.type, enabled=self.train_cfg.mixed_precision):
            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.train_cfg.clip_grad_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.train_cfg.clip_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach().item()), {
            k: float(v.detach().item()) for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def _validate(self) -> float:
        """Validate kept detections."""
        self.model.eval()
        total_score = 0.0
        total_count = 0
        for images, _ in self.val_loader:
            device_images = [img.to(self.device) for img in images]
            outputs = self.model(device_images)
            for out in outputs:
                scores = out.get("scores")
                if scores is not None:
                    mask = scores >= self.eval_cfg.score_threshold
                    total_score += float(scores[mask].sum().item())
                    total_count += int(mask.sum().item())
        return (total_score / max(total_count, 1)) if total_count > 0 else 0.0

    def fit(self) -> TrainArtifacts:
        """Train the model."""
        best_metric = -1.0
        best_epoch = -1
        last_ckpt = None
        best_path = self.ckpt_dir / "best.pt"

        for epoch in range(1, self.train_cfg.epochs + 1):
            running = 0.0
            for images, targets in self.train_loader:
                loss, _details = self._step(images, targets)
                running += loss
            avg_loss = running / max(len(self.train_loader), 1)

            val_metric = self._validate()
            logger.info(
                "Epoch %s/%s - loss: %.4f - val_metric: %.4f",
                epoch,
                self.train_cfg.epochs,
                avg_loss,
                val_metric,
            )

            # Save checkpoint
            ckpt = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_cfg": self.model_cfg.model_dump(),
                "epoch": epoch,
                "val_metric": val_metric,
            }
            ckpt_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            last_ckpt = ckpt_path

            if val_metric >= best_metric:
                best_metric = val_metric
                best_epoch = epoch
                torch.save(ckpt, best_path)

            if self.train_cfg.save_best_only and last_ckpt and ckpt_path != best_path:
                # optionally prune non-best checkpoints (keep latest and best)
                try:
                    if ckpt_path.exists():
                        ckpt_path.unlink()
                except OSError:
                    pass

        meta = {
            "data": {
                "images_dir": str(self.data_cfg.images_dir),
                "annotations_path": str(self.data_cfg.annotations_path),
            },
            "model": self.model_cfg.model_dump(),
            "train": self.train_cfg.model_dump(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        with (self.ckpt_dir / "training_summary.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return TrainArtifacts(
            checkpoint_path=(
                self.ckpt_dir / "best.pt"
                if self.train_cfg.save_best_only
                else last_ckpt
            ),
            best_epoch=best_epoch,
            best_metric=best_metric,
            params=count_parameters(self.model),
        )


class HFCocoDataset(torch.utils.data.Dataset):
    """Hugging Face COCO dataset."""

    def __init__(
        self, images_dir: Path, annotations_path: Path, split_ratio: float, split: str
    ) -> None:
        super().__init__()
        data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        self.images_dir = Path(images_dir)
        self.images = sorted(data.get("images", []), key=lambda im: im.get("id", 0))
        anns = data.get("annotations", [])
        self.anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in anns:
            self.anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        cutoff = int(len(self.images) * split_ratio)
        self.indices = (
            list(range(0, cutoff))
            if split == "train"
            else list(range(cutoff, len(self.images)))
        )

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from the dataset."""
        real_idx = self.indices[idx]
        image_info = self.images[real_idx]
        file_name = image_info.get("file_name")
        image_id = int(image_info.get("id"))
        image_path = self.images_dir / str(file_name)
        image = Image.open(image_path).convert("RGB")
        annotations = [
            {
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "category_id": int(ann.get("category_id", 1)),
                "area": float(ann.get("area", 0.0)),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            for ann in self.anns_by_image.get(image_id, [])
        ]
        return {"image": image, "image_id": image_id, "annotations": annotations}


@dataclass
class HFTrainArtifacts:
    """Artifacts from Hugging Face training."""

    output_dir: Path
    best_metrics: dict[str, Any]
    best_model_path: Path | None


class HFTrainer:
    """Hugging Face trainer."""

    def __init__(
        self,
        model_cfg: HuggingFaceModelConfig,
        train_cfg: HFTrainingConfig,
        images_dir: Path,
        annotations_path: Path,
        train_split: float = 0.8,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.train_split = train_split
        self._processor = None
        self._model = None

    def _build(self) -> Trainer:
        """Build the model."""
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_cfg.model_id,
            revision=self.model_cfg.revision,
            cache_dir=self.model_cfg.cache_dir,
        )
        if self.model_cfg.task == "detection":
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.revision,
                cache_dir=self.model_cfg.cache_dir,
                ignore_mismatched_sizes=self.model_cfg.num_labels is not None,
                num_labels=self.model_cfg.num_labels,
            )
        elif self.model_cfg.task == "instance_segmentation":
            self._model = AutoModelForInstanceSegmentation.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.revision,
                cache_dir=self.model_cfg.cache_dir,
                ignore_mismatched_sizes=self.model_cfg.num_labels is not None,
                num_labels=self.model_cfg.num_labels,
            )
        else:
            msg = "semantic_segmentation training not integrated"
            raise NotImplementedError(msg)

        train_ds = HFCocoDataset(
            self.images_dir, self.annotations_path, self.train_split, "train"
        )
        eval_ds = HFCocoDataset(
            self.images_dir, self.annotations_path, self.train_split, "val"
        )

        def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
            images = [e["image"] for e in examples]
            annotations = [e["annotations"] for e in examples]
            return self._processor(
                images=images, annotations=annotations, return_tensors="pt"
            )

        args = TrainingArguments(
            output_dir=str(self.train_cfg.output_dir),
            num_train_epochs=self.train_cfg.num_train_epochs,
            per_device_train_batch_size=self.train_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_cfg.per_device_eval_batch_size,
            learning_rate=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            lr_scheduler_type=self.train_cfg.lr_scheduler_type,
            warmup_ratio=self.train_cfg.warmup_ratio,
            fp16=self.train_cfg.fp16,
            logging_steps=self.train_cfg.logging_steps,
            evaluation_strategy=self.train_cfg.evaluation_strategy,
            save_strategy=self.train_cfg.save_strategy,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            load_best_model_at_end=self.train_cfg.evaluation_strategy != "no",
            seed=self.train_cfg.seed,
            report_to=[],
        )

        return Trainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds
            if self.train_cfg.evaluation_strategy != "no"
            else None,
            data_collator=collate_fn,
            tokenizer=self._processor,
        )

    def fit(self) -> HFTrainArtifacts:
        trainer = self._build()  # type: ignore[assignment]
        train_result = trainer.train()
        metrics = train_result.metrics or {}
        trainer.save_model()
        (self.train_cfg.output_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        best_path = None
        if getattr(trainer.args, "load_best_model_at_end", False):
            best_path = Path(trainer.args.output_dir) / "checkpoint-best"
        return HFTrainArtifacts(
            output_dir=self.train_cfg.output_dir,
            best_metrics=metrics,
            best_model_path=best_path,
        )
