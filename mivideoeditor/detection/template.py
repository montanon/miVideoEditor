"""Template-based detector with color pre-filtering."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from mivideoeditor.core.constants import SUPPORTED_AREA_TYPES
from mivideoeditor.core.models import BoundingBox, DetectionResult, SensitiveArea
from mivideoeditor.detection.base import BaseDetector, DetectionConfig, DetectionError
from mivideoeditor.detection.constants import (
    ATUIN_HIGHLIGHT_HSV_LOWER,
    ATUIN_HIGHLIGHT_HSV_UPPER,
    ATUIN_MAX_ASPECT_RATIO,
    ATUIN_MIN_ASPECT_RATIO,
    ATUIN_TERMINAL_HSV_LOWER,
    ATUIN_TERMINAL_HSV_UPPER,
    CHATGPT_DARK_HSV_LOWER,
    CHATGPT_DARK_HSV_UPPER,
    CHATGPT_LIGHT_HSV_LOWER,
    CHATGPT_LIGHT_HSV_UPPER,
    CHATGPT_MAX_ASPECT_RATIO,
    CHATGPT_MIN_ASPECT_RATIO,
    DEFAULT_IOU_THRESHOLD,
    GENERIC_MAX_ASPECT_RATIO,
    GENERIC_MIN_ASPECT_RATIO,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_CANDIDATE_REGIONS,
    MORPHOLOGY_KERNEL_SIZE,
    TEMPLATE_MIN_SIZE,
    TEMPLATE_PADDING_PIXELS,
    TEMPLATE_SCALE_TOLERANCE,
)
from mivideoeditor.storage.annotation_service import AnnotationService
from mivideoeditor.utils.image import ImageUtils

logger = logging.getLogger(__name__)


class ColorProfile:
    """Color profile for interface detection."""

    def __init__(
        self,
        area_type: str,
        hsv_lower: tuple[int, int, int],
        hsv_upper: tuple[int, int, int],
        name: str | None = None,
    ) -> None:
        self.area_type = area_type
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.name = name or f"{area_type}_profile"

    def create_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create binary mask from HSV image."""
        return cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)


class Detection:
    """Individual detection with metadata."""

    def __init__(
        self,
        bbox: BoundingBox,
        confidence: float,
        detection_type: str,
        template_match_score: float = 0.0,
        color_match_score: float = 0.0,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.detection_type = detection_type
        self.template_match_score = template_match_score
        self.color_match_score = color_match_score


class TemplateDetector(BaseDetector):
    """Template matching detector with color pre-filtering."""

    def __init__(self, config: DetectionConfig, area_type: str) -> None:
        super().__init__(config)

        if area_type not in SUPPORTED_AREA_TYPES:
            msg = (
                f"Unsupported area type: {area_type}. "
                f"Supported: {list(SUPPORTED_AREA_TYPES.keys())}"
            )
            raise ValueError(msg)

        self.area_type = area_type
        self.templates: dict[str, np.ndarray] = {}
        self.color_profiles: list[ColorProfile] = []
        self.template_scales = config.template_scales

        # Initialize default color profiles
        self._initialize_default_color_profiles()

    def _initialize_default_color_profiles(self) -> None:
        """Initialize default color profiles for known interfaces."""
        if self.area_type == "chatgpt":
            # ChatGPT interface color profiles
            # Dark sidebar/background
            self.color_profiles.append(
                ColorProfile(
                    area_type="chatgpt",
                    hsv_lower=CHATGPT_DARK_HSV_LOWER,
                    hsv_upper=CHATGPT_DARK_HSV_UPPER,
                    name="chatgpt_dark",
                )
            )
            # White chat bubbles
            self.color_profiles.append(
                ColorProfile(
                    area_type="chatgpt",
                    hsv_lower=CHATGPT_LIGHT_HSV_LOWER,
                    hsv_upper=CHATGPT_LIGHT_HSV_UPPER,
                    name="chatgpt_light",
                )
            )

        elif self.area_type == "atuin":
            # Atuin terminal interface
            # Dark terminal background
            self.color_profiles.append(
                ColorProfile(
                    area_type="atuin",
                    hsv_lower=ATUIN_TERMINAL_HSV_LOWER,
                    hsv_upper=ATUIN_TERMINAL_HSV_UPPER,
                    name="atuin_terminal",
                )
            )
            # Search highlight
            self.color_profiles.append(
                ColorProfile(
                    area_type="atuin",
                    hsv_lower=ATUIN_HIGHLIGHT_HSV_LOWER,
                    hsv_upper=ATUIN_HIGHLIGHT_HSV_UPPER,
                    name="atuin_highlight",
                )
            )

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Detect sensitive regions using template matching with color pre-filtering."""
        start_time = time.time()

        try:
            self._validate_frame(frame)

            if not self.is_trained and not self.templates:
                logger.warning("Detector not trained and no templates loaded")
                return DetectionResult(
                    detections=[],
                    detection_time=time.time() - start_time,
                    detector_type=f"template_{self.area_type}",
                    timestamp=timestamp,
                    frame_metadata={"warning": "detector_not_trained"},
                )

            # Stage 1: Color pre-filtering to find candidate regions
            candidate_regions = self._apply_color_filters(frame)

            if not candidate_regions:
                logger.debug("No candidate regions found in color filtering")
                detection_time = time.time() - start_time
                result = DetectionResult(
                    detections=[],
                    detection_time=detection_time,
                    detector_type=f"template_{self.area_type}",
                    timestamp=timestamp,
                    frame_metadata={"candidate_regions": 0},
                )
                self._update_stats(result)
                return result

            # Stage 2: Template matching within candidate regions
            detections = []
            for region in candidate_regions[:MAX_CANDIDATE_REGIONS]:
                region_detections = self._match_templates_in_region(frame, region)
                detections.extend(region_detections)

            # Stage 3: Post-processing
            filtered_detections = self._post_process_detections(detections)

            # Create result
            detection_time = time.time() - start_time
            detections_list = [
                (d.bbox, d.confidence, self.area_type) for d in filtered_detections
            ]
            result = DetectionResult(
                detections=detections_list,
                detection_time=detection_time,
                detector_type=f"template_{self.area_type}",
                timestamp=timestamp,
                frame_metadata={
                    "candidate_regions": len(candidate_regions),
                    "raw_detections": len(detections),
                    "filtered_detections": len(filtered_detections),
                },
            )

            self._update_stats(result)
        except Exception as e:
            logger.exception("Detection failed for frame at timestamp %s", timestamp)
            error_result = DetectionResult(
                detections=[],
                detection_time=time.time() - start_time,
                detector_type=f"template_{self.area_type}",
                timestamp=timestamp,
                frame_metadata={"error": str(e)},
            )
            self._update_stats(error_result)
            return error_result
        return result

    def _apply_color_filters(self, frame: np.ndarray) -> list[BoundingBox]:
        """Apply color filtering to find candidate regions."""
        if not self.color_profiles:
            logger.warning("No color profiles available")
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Combine all color masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for profile in self.color_profiles:
            mask = profile.create_mask(hsv)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPHOLOGY_KERNEL_SIZE)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidate_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_detection_area:
                continue

            if self.config.max_detection_area and area > self.config.max_detection_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Validate aspect ratio for interface elements
            aspect_ratio = w / h
            if not self._validate_aspect_ratio(aspect_ratio):
                continue

            # Add some padding around the region
            padding = TEMPLATE_PADDING_PIXELS
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)

            candidate_regions.append(BoundingBox(x, y, w, h))

        # Sort by area (larger regions first)
        candidate_regions.sort(key=lambda r: r.area, reverse=True)

        logger.debug("Found %d candidate regions", len(candidate_regions))
        return candidate_regions

    def _validate_aspect_ratio(self, aspect_ratio: float) -> bool:
        """Validate if aspect ratio matches expected interface proportions."""
        if self.area_type == "chatgpt":
            # ChatGPT interface is typically wider than tall
            return CHATGPT_MIN_ASPECT_RATIO <= aspect_ratio <= CHATGPT_MAX_ASPECT_RATIO
        if self.area_type == "atuin":
            # Atuin can be various ratios depending on terminal size
            return ATUIN_MIN_ASPECT_RATIO <= aspect_ratio <= ATUIN_MAX_ASPECT_RATIO

        # Generic validation
        return GENERIC_MIN_ASPECT_RATIO <= aspect_ratio <= GENERIC_MAX_ASPECT_RATIO

    def _match_templates_in_region(
        self, frame: np.ndarray, region: BoundingBox
    ) -> list[Detection]:
        """Perform template matching within a candidate region."""
        if not self.templates:
            return []

        # Extract region
        region_image = frame[
            region.y : region.y + region.height, region.x : region.x + region.width
        ]

        if region_image.size == 0:
            return []

        detections = []

        for template_name, template in self.templates.items():
            # Try multiple scales
            for scale in self.template_scales:
                scaled_template = self._scale_template(template, scale)

                if (
                    scaled_template.shape[0] > region_image.shape[0]
                    or scaled_template.shape[1] > region_image.shape[1]
                ):
                    continue

                # Template matching
                try:
                    match_score = ImageUtils.template_match(
                        region_image, scaled_template
                    )

                    if match_score > self.config.confidence_threshold:
                        # Calculate confidence based on match score
                        confidence = min(match_score, 1.0)

                        # Create detection (template size at region coordinates)
                        detection_bbox = BoundingBox(
                            x=region.x,
                            y=region.y,
                            width=scaled_template.shape[1],
                            height=scaled_template.shape[0],
                        )

                        detections.append(
                            Detection(
                                bbox=detection_bbox,
                                confidence=confidence,
                                detection_type=f"template_{template_name}",
                                template_match_score=match_score,
                            )
                        )

                except (ValueError, RuntimeError, cv2.error) as e:
                    logger.debug(
                        "Template matching failed for %s: %s", template_name, e
                    )
                    continue

        return detections

    def _scale_template(self, template: np.ndarray, scale: float) -> np.ndarray:
        """Scale template by given factor."""
        if abs(scale - 1.0) < TEMPLATE_SCALE_TOLERANCE:  # No scaling needed
            return template

        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)

        if new_width < TEMPLATE_MIN_SIZE or new_height < TEMPLATE_MIN_SIZE:
            return template

        return cv2.resize(
            template, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

    def _post_process_detections(self, detections: list[Detection]) -> list[Detection]:
        """Post-process detections to remove duplicates and apply filters."""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Non-maximum suppression to remove overlapping detections
        filtered_detections = []

        for detection in detections:
            is_duplicate = False

            for existing in filtered_detections:
                iou = detection.bbox.iou(existing.bbox)
                if iou > DEFAULT_IOU_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(detection)

                # Limit number of detections per frame
                if len(filtered_detections) >= self.config.max_regions_per_frame:
                    break

        return filtered_detections

    def train(self, annotations: list[SensitiveArea]) -> None:
        """Train detector on annotated sensitive areas."""
        if not annotations:
            msg = "No annotations provided for training"
            raise ValueError(msg)

        logger.info("Training template detector with %d annotations", len(annotations))

        # Filter annotations for this area type
        relevant_annotations = [
            ann for ann in annotations if ann.area_type == self.area_type
        ]

        if not relevant_annotations:
            msg = f"No annotations found for area type '{self.area_type}'"
            raise ValueError(msg)

        try:
            # Extract templates from annotations
            self._extract_templates_from_annotations(relevant_annotations)

            # Learn color profiles
            self._learn_color_profiles_from_annotations(relevant_annotations)

            self.is_trained = True
            logger.info("Training completed successfully")

        except Exception as e:
            msg = f"Training failed: {e}"
            logger.exception(msg)
            raise DetectionError(msg) from e

    def train_from_storage(
        self, annotation_service: AnnotationService, video_id: str | None = None
    ) -> None:
        """Train detector using annotations from storage service."""
        try:
            if video_id:
                # Get annotations for specific video
                annotations = annotation_service.get_annotations_for_video(video_id)
                logger.info(
                    "Loading %d annotations for video %s", len(annotations), video_id
                )
            else:
                # Get all annotations of this area type
                # Note: This would require extending AnnotationService
                logger.warning(
                    "Training from all annotations not yet implemented - "
                    "use video_id parameter"
                )
                annotations = []

            if not annotations:
                video_msg = f"video {video_id}" if video_id else "this detector"
                msg = f"No annotations found for training {video_msg}"
                raise ValueError(msg)

            # Convert AnnotationRecord objects back to SensitiveArea objects
            sensitive_areas = []
            for ann_record in annotations:
                # Create SensitiveArea from AnnotationRecord
                metadata = ann_record.metadata or {}
                metadata.update(
                    {
                        "video_id": ann_record.video_id,
                        "frame_number": ann_record.frame_number,
                        "annotated_by": ann_record.annotated_by,
                    }
                )

                sensitive_area = SensitiveArea(
                    timestamp=ann_record.timestamp,
                    bounding_box=ann_record.bounding_box,
                    area_type=ann_record.area_type,
                    confidence=ann_record.confidence,
                    metadata=metadata,
                    id=ann_record.id,
                    image_path=ann_record.image_path,
                )
                sensitive_areas.append(sensitive_area)

            # Use the regular train method
            self.train(sensitive_areas)
            logger.info("Training from storage completed successfully")

        except DetectionError:
            raise
        except Exception as e:
            msg = f"Training from storage failed: {e}"
            logger.exception(msg)
            raise DetectionError(msg) from e

    def _extract_templates_from_annotations(
        self, annotations: list[SensitiveArea]
    ) -> None:
        """Extract REAL templates from annotated regions using stored frame images."""
        logger.info("Extracting templates from %d annotations", len(annotations))

        template_count = 0

        for annotation in annotations:
            # Skip annotations without real frame data
            if not annotation.image_path or not annotation.image_path.exists():
                logger.warning(
                    "Skipping annotation %s - no frame image at %s",
                    annotation.id,
                    annotation.image_path,
                )
                continue

            # Load the real frame image
            frame_image = cv2.imread(str(annotation.image_path))
            if frame_image is None:
                logger.warning("Failed to load frame image: %s", annotation.image_path)
                continue

            # Extract the annotated region from the frame
            bbox = annotation.bounding_box
            height, width = frame_image.shape[:2]

            # Ensure bounding box is within frame bounds
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(width, bbox.x + bbox.width)
            y2 = min(height, bbox.y + bbox.height)

            # Extract the region of interest
            if x2 > x1 and y2 > y1:
                roi = frame_image[y1:y2, x1:x2].copy()

                # Resize template to reasonable size for matching
                roi = self._normalize_template_size(roi)

                # Store the real template
                template_key = f"{self.area_type}_{annotation.id}"
                self.templates[template_key] = roi
                template_count += 1

                logger.debug(
                    "Extracted template %s: %dx%d from %s",
                    template_key,
                    roi.shape[1],
                    roi.shape[0],
                    annotation.image_path,
                )

        if template_count == 0:
            msg = (
                f"No valid templates extracted for {self.area_type}. "
                "Ensure annotations have valid image_path with stored frame images."
            )
            raise DetectionError(msg)

        logger.info(
            "Extracted %d REAL templates from %d annotations",
            template_count,
            len(annotations),
        )

    def _normalize_template_size(self, template: np.ndarray) -> np.ndarray:
        """Normalize template size for consistent matching."""
        height, width = template.shape[:2]

        # Set reasonable bounds for template size
        max_width = 200
        max_height = 150
        min_width = 20
        min_height = 15

        # Calculate scaling factor to fit within bounds
        if width > max_width or height > max_height:
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)

            new_width = max(min_width, int(width * scale))
            new_height = max(min_height, int(height * scale))

            template = cv2.resize(
                template, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            logger.debug(
                "Resized template from %dx%d to %dx%d",
                width,
                height,
                new_width,
                new_height,
            )

        return template

    def _learn_color_profiles_from_annotations(
        self, annotations: list[SensitiveArea]
    ) -> None:
        """Learn and refine color profiles from annotated regions."""
        # Analyze annotation patterns to potentially refine color profiles
        logger.info(
            "Analyzing %d annotations for color profile refinement", len(annotations)
        )

        # Count annotations by confidence to understand detection quality
        high_conf_count = sum(
            1 for ann in annotations if ann.confidence > HIGH_CONFIDENCE_THRESHOLD
        )
        low_conf_count = sum(
            1 for ann in annotations if ann.confidence < LOW_CONFIDENCE_THRESHOLD
        )

        logger.info(
            "Annotation confidence distribution: %d high (>0.9), %d low (<0.6)",
            high_conf_count,
            low_conf_count,
        )

        # For now, keep the default profiles but log the analysis
        # In a future implementation, this could analyze actual frame colors
        # to refine the HSV ranges based on successful detections
        logger.info("Using default color profiles for %s", self.area_type)

    def save_model(self, path: Path) -> None:
        """Save trained templates and color profiles."""
        # Create model directory
        model_dir = path.parent / f"{path.stem}_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save base detector state
        super().save_model(path)

        # Save templates
        for name, template in self.templates.items():
            template_path = model_dir / f"template_{name}.npy"
            np.save(template_path, template)

        # Save color profiles
        profiles_data = [
            {
                "area_type": profile.area_type,
                "hsv_lower": profile.hsv_lower.tolist(),
                "hsv_upper": profile.hsv_upper.tolist(),
                "name": profile.name,
            }
            for profile in self.color_profiles
        ]

        profiles_path = model_dir / "color_profiles.json"
        with profiles_path.open("w", encoding="utf-8") as f:
            json.dump(profiles_data, f, indent=2)

        logger.info("Model saved to %s", model_dir)

    def load_model(self, path: Path) -> None:
        """Load trained templates and color profiles."""
        # Load base detector state
        super().load_model(path)

        model_dir = path.parent / f"{path.stem}_model"
        if not model_dir.exists():
            logger.warning("Model directory not found: %s", model_dir)
            return

        # Load templates
        self.templates = {}
        for template_file in model_dir.glob("template_*.npy"):
            template_name = template_file.stem.replace("template_", "")
            template = np.load(template_file)
            self.templates[template_name] = template

        # Load color profiles
        profiles_path = model_dir / "color_profiles.json"
        if profiles_path.exists():
            with profiles_path.open("r", encoding="utf-8") as f:
                profiles_data = json.load(f)

            self.color_profiles = []
            for profile_data in profiles_data:
                profile = ColorProfile(
                    area_type=profile_data["area_type"],
                    hsv_lower=tuple(profile_data["hsv_lower"]),
                    hsv_upper=tuple(profile_data["hsv_upper"]),
                    name=profile_data["name"],
                )
                self.color_profiles.append(profile)

        logger.info("Model loaded from %s", model_dir)
        logger.info(
            "Loaded %d templates and %d color profiles",
            len(self.templates),
            len(self.color_profiles),
        )
