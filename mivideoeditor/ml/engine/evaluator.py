"""Decoupled evaluation workflow for detection models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers import (
    AutoImageProcessor,
    AutoModelForInstanceSegmentation,
    AutoModelForObjectDetection,
)

from mivideoeditor.ml.config import TorchEvalConfig
from mivideoeditor.ml.engine.trainer import (
    HFCocoDataset,
    HuggingFaceModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Evaluation report for a model."""

    precision: float
    recall: float
    f1: float
    avg_score: float
    samples: int
    details: dict[str, Any]


class Evaluator:
    """Evaluator computes lightweight metrics to keep workflow independent."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        cfg: TorchEvalConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.cfg = cfg
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
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> EvalReport:
        """Evaluate the model."""
        self.model.eval()
        tp = 0
        fp = 0
        fn = 0
        total_score = 0.0
        total_count = 0

        for images, targets in self.loader:
            device_images = [img.to(self.device) for img in images]
            outputs = self.model(device_images)

            for out, tgt in zip(outputs, targets, strict=True):
                scores = out.get("scores", torch.tensor([]))
                boxes_p = out.get("boxes", torch.empty((0, 4)))
                labels_p = out.get("labels", torch.empty((0,), dtype=torch.int64))

                mask = scores >= self.cfg.score_threshold
                boxes_p = boxes_p[mask]
                labels_p = labels_p[mask]
                scores = scores[mask]

                gt_boxes = tgt.get("boxes", torch.empty((0, 4))).to(self.device)
                gt_labels = tgt.get("labels", torch.empty((0,), dtype=torch.int64)).to(
                    self.device
                )

                if gt_boxes.numel() == 0 and boxes_p.numel() == 0:
                    continue

                # Match predictions to GT by IoU and label
                ious = (
                    box_iou(boxes_p, gt_boxes)
                    if gt_boxes.numel() > 0
                    else torch.empty((0, 0))
                )

                used_gt = set()
                for i in range(boxes_p.size(0)):
                    j = -1
                    iou_max = 0.0
                    for k in range(gt_boxes.size(0)):
                        if k in used_gt:
                            continue
                        iou = float(ious[i, k]) if ious.numel() > 0 else 0.0
                        if (
                            iou > iou_max
                            and iou >= self.cfg.iou_threshold
                            and labels_p[i] == gt_labels[k]
                        ):
                            iou_max = iou
                            j = k
                    if j >= 0:
                        tp += 1
                        used_gt.add(j)
                        total_score += float(scores[i].item())
                        total_count += 1
                    else:
                        fp += 1

                fn += max(0, gt_boxes.size(0) - len(used_gt))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            (2 * precision * recall / max(precision + recall, 1e-8))
            if (precision + recall) > 0
            else 0.0
        )
        avg_score = total_score / max(total_count, 1) if total_count > 0 else 0.0
        samples = len(self.loader.dataset) if hasattr(self.loader, "dataset") else 0

        return EvalReport(
            precision=precision,
            recall=recall,
            f1=f1,
            avg_score=avg_score,
            samples=samples,
            details={"tp": tp, "fp": fp, "fn": fn},
        )


@dataclass
class HFEvalReport:
    """Hugging Face evaluation report."""

    precision: float
    recall: float
    f1: float
    samples: int
    details: dict[str, Any]


class HFEvaluator:
    """Hugging Face evaluator."""

    def __init__(
        self,
        model_cfg: HuggingFaceModelConfig,
        images_dir: Path,
        annotations_path: Path,
        split: str = "val",
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model_cfg = model_cfg
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.split = split
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._processor = None
        self._model = None

    def _setup(self) -> None:
        """Set up the evaluator."""
        self._box_iou = box_iou
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
            )
        elif self.model_cfg.task == "instance_segmentation":
            self._model = AutoModelForInstanceSegmentation.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.revision,
                cache_dir=self.model_cfg.cache_dir,
            )
        else:
            msg = "semantic_segmentation evaluation not integrated"
            raise NotImplementedError(msg)
        self._model.eval()

    def evaluate(self) -> HFEvalReport:
        """Evaluate the model."""
        ds = HFCocoDataset(self.images_dir, self.annotations_path, 0.8, self.split)
        if self._model is None:
            self._setup()
        tp = fp = fn = 0
        for i in range(len(ds)):
            ex = ds[i]
            image = ex["image"]
            gt = ex["annotations"]
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            post = self._processor.post_process_object_detection(
                outputs, threshold=self.score_threshold, target_sizes=target_sizes
            )[0]
            pred_boxes = post.get("boxes", torch.empty((0, 4)))
            pred_labels = post.get("labels", torch.empty((0,), dtype=torch.int64))
            gt_boxes = torch.tensor([ann["bbox"] for ann in gt], dtype=torch.float32)
            if gt_boxes.numel() > 0:
                gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
            gt_labels = torch.tensor(
                [ann["category_id"] for ann in gt], dtype=torch.int64
            )
            if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
                continue
            ious = (
                self._box_iou(pred_boxes, gt_boxes)
                if gt_boxes.numel() > 0
                else torch.empty((0, 0))
            )
            used_gt: set[int] = set()
            for i in range(pred_boxes.size(0)):
                best_j = -1
                best_iou = 0.0
                for j in range(gt_boxes.size(0)):
                    if j in used_gt:
                        continue
                    iou = float(ious[i, j]) if ious.numel() > 0 else 0.0
                    if (
                        iou >= self.iou_threshold
                        and int(pred_labels[i]) == int(gt_labels[j])
                        and iou > best_iou
                    ):
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    tp += 1
                    used_gt.add(best_j)
                else:
                    fp += 1
            fn += max(0, gt_boxes.size(0) - len(used_gt))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            (2 * precision * recall / max(precision + recall, 1e-8))
            if (precision + recall) > 0
            else 0.0
        )
        return HFEvalReport(
            precision=precision,
            recall=recall,
            f1=f1,
            samples=len(ds),
            details={"tp": tp, "fp": fp, "fn": fn},
        )
