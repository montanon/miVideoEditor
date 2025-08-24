"""Image utilities for processing, analysis, and similarity calculations."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mivideoeditor.core.models import BoundingBox, ValidationResult

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utilities for image processing, analysis, and similarity calculations."""

    @staticmethod
    def load_image(image_path: Path, color_mode: str = "BGR") -> np.ndarray:
        """Load an image file into a numpy array."""
        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # OpenCV color mode flags
        color_flags = {
            "BGR": cv2.IMREAD_COLOR,
            "RGB": cv2.IMREAD_COLOR,
            "GRAY": cv2.IMREAD_GRAYSCALE,
            "UNCHANGED": cv2.IMREAD_UNCHANGED,
        }

        if color_mode not in color_flags:
            msg = (
                f"Invalid color_mode: {color_mode}. Must be one of "
                f"{list(color_flags.keys())}"
            )
            raise ValueError(msg)

        try:
            image = cv2.imread(str(image_path), color_flags[color_mode])
            if image is None:
                msg = f"Failed to load image: {image_path}"
                logger.exception(msg)
                raise ValueError(msg)

            # Convert BGR to RGB if requested
            if color_mode == "RGB" and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            msg = f"Loaded image: {image_path}, shape: {image.shape}"
            logger.debug(msg)

        except Exception as e:
            msg = f"Error loading image {image_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return image

    @staticmethod
    def save_image(image: np.ndarray, output_path: Path, quality: int = 95) -> None:
        """Save a numpy array as an image file."""
        if image.size == 0:
            msg = "Cannot save empty image"
            raise ValueError(msg)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine file extension and encoding parameters
        ext = output_path.suffix.lower()
        encode_params = []

        if ext in [".jpg", ".jpeg"]:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
        elif ext == ".png":
            # PNG compression level (0-9, where 9 is maximum compression)
            compression = max(0, min(9, (100 - quality) // 11))
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext == ".webp":
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, max(0, min(100, quality))]

        try:
            success = cv2.imwrite(str(output_path), image, encode_params)
            if not success:
                msg = f"Failed to save image to: {output_path}"
                logger.error(msg)
                raise RuntimeError(msg)
            msg = f"Saved image to: {output_path}"
            logger.debug(msg)

        except Exception as e:
            msg = f"Error saving image to {output_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def resize_image(
        image: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        interpolation: str = "INTER_LINEAR",
        *,
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """Resize an image to specified dimensions."""
        if image.size == 0:
            msg = "Cannot resize empty image"
            raise ValueError(msg)

        original_height, original_width = image.shape[:2]

        # Interpolation methods
        interp_methods = {
            "INTER_NEAREST": cv2.INTER_NEAREST,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
        }

        if interpolation not in interp_methods:
            msg = (
                f"Invalid interpolation: {interpolation}. Must be one of "
                f"{list(interp_methods.keys())}"
            )
            raise ValueError(msg)

        # Calculate target dimensions
        if width is None and height is None:
            msg = "Must specify at least one dimension (width or height)"
            raise ValueError(msg)

        if maintain_aspect_ratio:
            if width is not None and height is not None:
                # Use the smaller scaling factor to maintain aspect ratio
                scale_w = width / original_width
                scale_h = height / original_height
                scale = min(scale_w, scale_h)
                target_width = int(original_width * scale)
                target_height = int(original_height * scale)
            elif width is not None:
                scale = width / original_width
                target_width = width
                target_height = int(original_height * scale)
            else:  # height is not None
                scale = height / original_height
                target_width = int(original_width * scale)
                target_height = height
        else:
            target_width = width or original_width
            target_height = height or original_height

        if target_width <= 0 or target_height <= 0:
            msg = f"Invalid target dimensions: {target_width}x{target_height}"
            raise ValueError(msg)

        try:
            resized = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=interp_methods[interpolation],
            )
            msg = (
                f"Resized image from {original_width}x{original_height} to "
                f"{target_width}x{target_height}"
            )
            logger.debug(msg)

        except Exception as e:
            msg = f"Error resizing image: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return resized

    @staticmethod
    def crop_image(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Crop an image using a bounding box."""
        if image.size == 0:
            msg = "Cannot crop empty image"
            raise ValueError(msg)

        height, width = image.shape[:2]

        # Validate bounding box
        if bbox.x < 0 or bbox.y < 0:
            msg = f"Bounding box coordinates must be non-negative: {bbox}"
            raise ValueError(msg)

        if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
            msg = f"Bounding box exceeds image dimensions: {bbox} vs {width}x{height}"
            raise ValueError(msg)

        try:
            cropped = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
            msg = f"Cropped image region: {bbox}"
            logger.debug(msg)
            return cropped.copy()  # Return a copy to avoid memory issues

        except Exception as e:
            msg = f"Error cropping image: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def calculate_image_hash(image: np.ndarray, hash_type: str = "sha256") -> str:
        """Calculate hash of an image for comparison purposes."""
        if image.size == 0:
            msg = "Cannot hash empty image"
            raise ValueError(msg)

        supported_hashes = {"sha256"}
        if hash_type not in supported_hashes:
            msg = (
                f"Unsupported hash type: {hash_type}. Must be one of {supported_hashes}"
            )
            raise ValueError(msg)

        try:
            # Convert image to bytes
            image_bytes = image.tobytes()

            # Create hash
            hash_obj = hashlib.sha256(image_bytes)

            return hash_obj.hexdigest()

        except Exception as e:
            msg = f"Error calculating image hash: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def calculate_structural_similarity(
        image1: np.ndarray, image2: np.ndarray
    ) -> float:
        """Calculate structural similarity (SSIM) between two images."""
        if image1.size == 0 or image2.size == 0:
            msg = "Cannot compare empty images"
            raise ValueError(msg)

        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Resize images to same dimensions if different
        if image1.shape != image2.shape:
            h, w = (
                min(image1.shape[0], image2.shape[0]),
                min(image1.shape[1], image2.shape[1]),
            )
            image1 = cv2.resize(image1, (w, h))
            image2 = cv2.resize(image2, (w, h))

        try:
            # Calculate SSIM using OpenCV's built-in function if available
            if hasattr(cv2, "quality") and hasattr(cv2.quality, "QualitySSIM_compute"):
                ssim_map = cv2.quality.QualitySSIM_compute(image1, image2)
                return float(ssim_map[0])
            return ImageUtils._compute_ssim_manual(image1, image2)

        except Exception as e:
            msg = f"Error calculating SSIM: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def _compute_ssim_manual(image1: np.ndarray, image2: np.ndarray) -> float:
        """Manually compute SSIM when OpenCV quality module is not available."""
        # Constants for SSIM calculation
        ssim_c1 = 0.01**2
        ssim_c2 = 0.03**2

        # Convert to float
        img1 = image1.astype(np.float64)
        img2 = image2.astype(np.float64)

        # Calculate means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        # Calculate SSIM
        numerator1 = 2 * mu1_mu2 + ssim_c1
        numerator2 = 2 * sigma12 + ssim_c2
        denominator1 = mu1_sq + mu2_sq + ssim_c1
        denominator2 = sigma1_sq + sigma2_sq + ssim_c2

        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        return float(np.mean(ssim_map))

    @staticmethod
    def calculate_histogram_similarity(
        image1: np.ndarray, image2: np.ndarray, method: str = "correlation"
    ) -> float:
        """Calculate histogram similarity between two images."""
        if image1.size == 0 or image2.size == 0:
            msg = "Cannot compare empty images"
            raise ValueError(msg)

        methods = {
            "correlation": cv2.HISTCMP_CORREL,
            "chi_square": cv2.HISTCMP_CHISQR,
            "intersection": cv2.HISTCMP_INTERSECT,
            "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
        }

        if method not in methods:
            msg = f"Invalid method: {method}. Must be one of {list(methods.keys())}"
            raise ValueError(msg)

        try:
            # Convert to grayscale for histogram calculation
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # Calculate histograms
            hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

            # Normalize histograms
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Calculate similarity
            similarity = cv2.compareHist(hist1, hist2, methods[method])

            # Convert to 0-1 range for consistency
            if method in ["chi_square", "bhattacharyya"]:
                # Lower values mean higher similarity, so invert
                similarity = 1.0 / (1.0 + similarity)

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            msg = f"Error calculating histogram similarity: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def find_template_matches(
        source_image: np.ndarray,
        template_image: np.ndarray,
        threshold: float = 0.8,
        method: str = "TM_CCOEFF_NORMED",
    ) -> list[dict[str, Any]]:
        """Find template matches in a source image."""
        if source_image.size == 0 or template_image.size == 0:
            msg = "Cannot match with empty images"
            raise ValueError(msg)

        # Template matching methods
        methods = {
            "TM_CCOEFF": cv2.TM_CCOEFF,
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR": cv2.TM_CCORR,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF": cv2.TM_SQDIFF,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        }

        if method not in methods:
            msg = f"Invalid method: {method}. Must be one of {list(methods.keys())}"
            raise ValueError(msg)

        if not (0.0 <= threshold <= 1.0):
            msg = f"Threshold must be between 0.0 and 1.0, got {threshold}"
            raise ValueError(msg)

        try:
            # Convert to grayscale for template matching
            source_gray = source_image
            template_gray = template_image

            if len(source_image.shape) == 3:
                source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            if len(template_image.shape) == 3:
                template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

            # Get template dimensions
            template_h, template_w = template_gray.shape

            # Perform template matching
            result = cv2.matchTemplate(source_gray, template_gray, methods[method])

            # Find locations above threshold
            matches = []

            if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                # For squared difference methods, lower values are better matches
                locations = np.where(result <= (1.0 - threshold))
                match_values = result[locations]
                # Convert to similarity score (invert)
                confidences = 1.0 - match_values
            else:
                # For other methods, higher values are better matches
                locations = np.where(result >= threshold)
                confidences = result[locations]

            # Create match dictionaries
            for i, (y, x) in enumerate(zip(*locations, strict=True)):
                matches.append(
                    {
                        "bbox": BoundingBox(
                            x=int(x), y=int(y), width=template_w, height=template_h
                        ),
                        "confidence": float(confidences[i]),
                        "center": (int(x + template_w // 2), int(y + template_h // 2)),
                        "method": method,
                    }
                )

            # Sort by confidence (highest first)
            matches.sort(key=lambda m: m["confidence"], reverse=True)
            msg = f"Found {len(matches)} template matches above threshold {threshold}"
            logger.debug(msg)

        except Exception as e:
            msg = f"Error finding template matches: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return matches

    @staticmethod
    def validate_image_file(image_path: Path) -> ValidationResult:
        """Validate an image file for processing compatibility."""
        result = ValidationResult(is_valid=True)

        # Basic file validation
        if not image_path.exists():
            result.add_error(f"Image file not found: {image_path}")
            return result

        if not image_path.is_file():
            result.add_error(f"Path is not a file: {image_path}")
            return result

        # File extension check
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        }
        if image_path.suffix.lower() not in supported_extensions:
            result.add_warning(f"Unusual image extension: {image_path.suffix}")

        # File size checks
        try:
            file_size = image_path.stat().st_size

            if file_size == 0:
                result.add_error("Image file is empty")
                return result
            if file_size < 100:  # Less than 100 bytes
                result.add_warning(f"Very small image file: {file_size} bytes")
            if file_size > 100 * 1024 * 1024:  # Greater than 100MB
                result.add_warning(
                    f"Very large image file: {file_size / (1024 * 1024):.1f}MB"
                )

        except OSError as e:
            result.add_error(f"Cannot access file: {e}")
            return result

        # Try to load and validate image
        try:
            image = ImageUtils.load_image(image_path)

            # Store basic image info in context
            height, width = image.shape[:2]
            result.context["width"] = int(width)
            result.context["height"] = int(height)
            result.context["channels"] = int(
                image.shape[2] if len(image.shape) == 3 else 1
            )
            result.context["megapixels"] = round((width * height) / 1_000_000, 2)

            # Validate dimensions
            if width < 16 or height < 16:
                result.add_warning(f"Very small image dimensions: {width}x{height}")
            if width > 8192 or height > 8192:
                result.add_warning(f"Very large image dimensions: {width}x{height}")

            # Validate aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                result.add_warning(f"Unusual aspect ratio: {aspect_ratio:.2f}")

        except (FileNotFoundError, RuntimeError, ValueError) as e:
            result.add_error(f"Invalid image file: {e}")

        return result

    @staticmethod
    def get_image_info(image_path: Path) -> dict[str, Any]:
        """Get comprehensive information about an image file."""
        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            raise FileNotFoundError(msg)

        try:
            image = ImageUtils.load_image(image_path, "UNCHANGED")
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            file_size = image_path.stat().st_size

            return {
                "file_path": str(image_path),
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "dtype": str(image.dtype),
                "file_size": int(file_size),
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "megapixels": round((width * height) / 1_000_000, 2),
                "aspect_ratio": round(width / height, 2),
                "color_mode": "grayscale" if channels == 1 else f"{channels}-channel",
            }

        except Exception as e:
            msg = f"Error getting image info for {image_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def create_color_mask(
        image: np.ndarray,
        lower_color: tuple[int, int, int],
        upper_color: tuple[int, int, int],
        color_space: str = "HSV",
    ) -> np.ndarray:
        """Create a binary mask based on color range."""
        if image.size == 0:
            msg = "Cannot create mask from empty image"
            raise ValueError(msg)

        color_spaces = {
            "HSV": cv2.COLOR_BGR2HSV,
            "LAB": cv2.COLOR_BGR2LAB,
            "RGB": cv2.COLOR_BGR2RGB,
            "BGR": None,  # No conversion needed
        }

        if color_space not in color_spaces:
            msg = (
                f"Invalid color_space: {color_space}. Must be one of "
                f"{list(color_spaces.keys())}"
            )
            raise ValueError(msg)

        try:
            # Convert color space if needed
            if color_spaces[color_space] is not None:
                converted_image = cv2.cvtColor(image, color_spaces[color_space])
            else:
                converted_image = image

            # Create mask
            mask = cv2.inRange(converted_image, lower_color, upper_color)
            msg = f"Created color mask in {color_space} space"
            logger.debug(msg)

        except Exception as e:
            msg = f"Error creating color mask: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return mask
