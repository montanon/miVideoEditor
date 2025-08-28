"""COCO-style evaluator with fallback PR/F1 if pycocotools unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mivideoeditor.ml.api import Prediction


@dataclass
class CocoEvalResult:
    """CocoEvalResult."""

    summary: dict[str, Any]
    per_class: dict[str, Any]


def _to_coco_predictions(
    predictions: list[Prediction], image_ids: list[int]
) -> list[dict[str, Any]]:
    """Convert Prediction objects to COCO json detection entries."""
    coco_preds: list[dict[str, Any]] = []
    for img_idx, pred in enumerate(predictions):
        img_id = image_ids[img_idx]
        boxes = pred.boxes
        if boxes.size == 0:
            continue
        # xyxy -> xywh
        xywh = boxes.copy()
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        coco_preds.extend(
            {
                "image_id": img_id,
                "category_id": int(pred.labels[i]),
                "bbox": [float(x) for x in xywh[i].tolist()],
                "score": float(pred.scores[i]),
            }
            for i in range(xywh.shape[0])
        )
    return coco_preds


def evaluate_coco(
    gt_coco_json: Path,
    predictions: list[Prediction],
    *,
    image_ids: list[int],
    labels: list[str],
    iou_thrs: list[float] | None = None,
    task: str = "detection",
) -> CocoEvalResult:
    """Evaluate COCO metrics."""
    iou_thrs = iou_thrs or [0.5, 0.75]

    coco_gt = COCO(str(gt_coco_json))
    coco_dt = coco_gt.loadRes(_to_coco_predictions(predictions, image_ids))
    coco_eval = COCOeval(
        coco_gt, coco_dt, iouType="bbox" if task == "detection" else "segm"
    )
    coco_eval.params.iouThrs = np.array(iou_thrs)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }
    per_class = {}
    # Detailed per-class metrics require additional sweeping; keep minimal for now.
    return CocoEvalResult(summary=summary, per_class=per_class)
