"""Data utilities and datasets for ML pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision.transforms import v2 as T

from mivideoeditor.ml.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class CocoRecord:
    """Coco-like record."""

    image_path: Path
    boxes_xywh: list[list[float]]
    labels: list[int]
    image_id: int


def _load_coco_like(annotations_path: Path, images_dir: Path) -> list[CocoRecord]:
    """Load a minimal COCO-like dataset into memory."""
    with Path(annotations_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img.get("file_name") for img in data.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    records: list[CocoRecord] = []
    for image_id, filename in id_to_filename.items():
        if filename is None:
            continue
        image_path = images_dir / filename
        anns = anns_by_image.get(image_id, [])
        boxes = [ann["bbox"] for ann in anns if "bbox" in ann]
        labels = [int(ann.get("category_id", 1)) for ann in anns]
        records.append(
            CocoRecord(
                image_path=image_path,
                boxes_xywh=boxes,
                labels=labels,
                image_id=image_id,
            )
        )
    logger.info("Loaded %d images from COCO-like annotations", len(records))
    return records


class DetectionDataset(Dataset):
    """Torch dataset for object detection from COCO-like JSON."""

    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.records = _load_coco_like(cfg.annotations_path, cfg.images_dir)

        # deterministic split by image_id
        self.records.sort(key=lambda r: r.image_id)
        idx = int(len(self.records) * cfg.train_split)
        if split == "train":
            self.records = self.records[:idx]
        else:
            self.records = self.records[idx:]

        size = cfg.image_size
        base_transforms: list[Callable] = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size, max_size=size),
        ]
        if cfg.augment and split == "train":
            base_transforms.extend(
                [
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
        self.transforms = T.Compose(base_transforms)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Get an item from the dataset."""
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        image = self.transforms(image)

        boxes_xyxy = []
        if rec.boxes_xywh:
            boxes_xyxy = box_convert(
                torch.tensor(rec.boxes_xywh, dtype=torch.float32),
                in_fmt="xywh",
                out_fmt="xyxy",
            )

        labels = torch.tensor(rec.labels or [], dtype=torch.int64)
        target = {
            "boxes": boxes_xyxy
            if len(rec.boxes_xywh) > 0
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": labels,
            "image_id": torch.tensor([rec.image_id], dtype=torch.int64),
        }
        return image, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[list[torch.Tensor], list[dict]]:
    """Collate a batch of items into a batch of tensors and targets."""
    images, targets = list(zip(*batch, strict=True))
    return list(images), list(targets)
