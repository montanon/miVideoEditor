"""Utilities to convert repo annotations into COCO format."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mivideoeditor.core.models import SensitiveArea, Timeline

logger = logging.getLogger(__name__)


@dataclass
class Category:
    """Category for a COCO dataset."""

    id: int
    name: str


def sensitive_areas_to_coco(
    areas: Iterable[SensitiveArea],
    *,
    categories: dict[str, int],
    output_path: Path,
) -> Path:
    """Convert a list of SensitiveArea to a COCO detection JSON."""
    images_index: dict[str, int] = {}
    images = []
    annotations = []

    ann_id = 1
    for area in areas:
        if area.image_path is None:
            logger.warning("Skipping area without image_path: %s", area)
            continue
        img_path = str(Path(area.image_path))
        if img_path not in images_index:
            try:
                with Image.open(area.image_path) as im:
                    width, height = im.size
            except (OSError, ValueError, RuntimeError):
                width, height = area.bounding_box.x2, area.bounding_box.y2
            image_id = len(images) + 1
            images_index[img_path] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": Path(img_path).name,
                    "width": width,
                    "height": height,
                }
            )

        image_id = images_index[img_path]
        bbox = area.bounding_box
        coco_bbox = [bbox.x, bbox.y, bbox.width, bbox.height]
        cat_id = int(categories.get(area.area_type, categories.get("custom", 1)))
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "bbox": coco_bbox,
                "category_id": cat_id,
                "iscrowd": 0,
                "area": bbox.area,
            }
        )
        ann_id += 1

    cats = [
        {"id": cid, "name": name}
        for name, cid in sorted(categories.items(), key=lambda x: x[1])
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": cats,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    logger.info(
        "Wrote COCO annotations to %s (%d images, %d anns)",
        output_path,
        len(images),
        len(annotations),
    )
    return output_path


def timeline_to_coco(
    timeline: Timeline,
    frame_images: dict[float, Path],
    *,
    categories: dict[str, int],
    output_path: Path,
) -> Path:
    """Convert a Timeline to COCO by sampling provided frame images."""
    images_index: dict[str, int] = {}
    images = []
    annotations = []
    ann_id = 1

    for ts, img_path in frame_images.items():
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except (OSError, ValueError, RuntimeError):
            width, height = 1920, 1080  # fallback
        image_id = len(images) + 1
        images_index[str(img_path)] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": Path(img_path).name,
                "width": width,
                "height": height,
                "timestamp": ts,
            }
        )

    sorted_frames = sorted(frame_images.items(), key=lambda kv: kv[0])
    frame_ts = [ts for ts, _ in sorted_frames]
    frame_paths = [path for _, path in sorted_frames]

    def _closest_frame(t: float) -> Path:
        idx = min(range(len(frame_ts)), key=lambda i: abs(frame_ts[i] - t))
        return frame_paths[idx]

    for region in timeline.blur_regions:
        img_path = _closest_frame(region.start_time)
        image_id = images_index[str(img_path)]
        bbox = region.bounding_box
        coco_bbox = [bbox.x, bbox.y, bbox.width, bbox.height]
        cat_name = str(region.metadata.get("area_type", "custom"))
        cat_id = int(categories.get(cat_name, categories.get("custom", 1)))

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "bbox": coco_bbox,
                "category_id": cat_id,
                "iscrowd": 0,
                "area": bbox.area,
            }
        )
        ann_id += 1

    cats = [
        {"id": cid, "name": name}
        for name, cid in sorted(categories.items(), key=lambda x: x[1])
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": cats,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    logger.info(
        "Wrote Timeline COCO to %s (%d images, %d anns)",
        output_path,
        len(images),
        len(annotations),
    )
    return output_path
