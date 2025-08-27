"""Decoupled training workflow for detection models."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mivideoeditor.ml.config import DataConfig, EvalConfig, ModelConfig, TrainConfig
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


class Trainer:
    """Trainer encapsulates dataset, model, loop and checkpointing."""

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

    def _step(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]):
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
            cpu_images = [img.to("cpu") for img in images]
            outputs = self.model(cpu_images)
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
                best_path = self.ckpt_dir / "best.pt"
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
            params=asdict(count_parameters(self.model)),
        )
