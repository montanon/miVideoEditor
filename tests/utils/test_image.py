"""Tests for image utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from mivideoeditor.core.models import BoundingBox
from mivideoeditor.utils.image import ImageUtils


class TestImageUtils:
    """Test image utilities."""

    def test_load_image_success(self, tmp_path: Path):
        """Test successful image loading."""
        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.touch()  # Create the file so it exists

        # Create a simple 3-channel test image
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_image[25:75, 25:75] = [255, 0, 0]  # Red square

        with patch("cv2.imread") as mock_imread:
            mock_imread.return_value = fake_image

            result = ImageUtils.load_image(test_image, color_mode="BGR")

            assert result is not None
            assert result.shape == (100, 100, 3)
            mock_imread.assert_called_once()

    def test_load_image_rgb_conversion(self, tmp_path: Path):
        """Test image loading with RGB conversion."""
        test_image = tmp_path / "test.jpg"
        test_image.touch()  # Create the file so it exists

        # BGR image (Blue, Green, Red channels)
        bgr_image = np.zeros((50, 50, 3), dtype=np.uint8)
        bgr_image[:, :] = [255, 0, 0]  # Blue in BGR

        with patch("cv2.imread") as mock_imread, patch("cv2.cvtColor") as mock_cvtcolor:
            mock_imread.return_value = bgr_image
            rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
            rgb_image[:, :] = [0, 0, 255]  # Red in RGB
            mock_cvtcolor.return_value = rgb_image

            ImageUtils.load_image(test_image, color_mode="RGB")

            mock_cvtcolor.assert_called_once_with(bgr_image, 4)  # COLOR_BGR2RGB = 4

    def test_load_image_grayscale(self, tmp_path: Path):
        """Test grayscale image loading."""
        test_image = tmp_path / "test.jpg"
        test_image.touch()  # Create the file so it exists

        gray_image = np.zeros((50, 50), dtype=np.uint8)
        gray_image[10:40, 10:40] = 128

        with patch("cv2.imread") as mock_imread:
            mock_imread.return_value = gray_image

            result = ImageUtils.load_image(test_image, color_mode="GRAY")

            assert result.shape == (50, 50)
            mock_imread.assert_called_once_with(
                str(test_image), 0
            )  # IMREAD_GRAYSCALE = 0

    def test_load_image_file_not_found(self) -> None:
        """Test image loading with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            ImageUtils.load_image(Path("nonexistent.jpg"))

    def test_load_image_invalid_color_mode(self, tmp_path: Path) -> None:
        """Test image loading with invalid color mode."""
        test_image = tmp_path / "test.jpg"
        test_image.touch()  # Create the file so it exists

        with pytest.raises(ValueError, match="Invalid color_mode"):
            ImageUtils.load_image(test_image, color_mode="INVALID")

    def test_load_image_opencv_fails(self, tmp_path: Path) -> None:
        """Test image loading when OpenCV fails."""
        test_image = tmp_path / "corrupted.jpg"
        test_image.touch()  # Create the file so it exists

        with patch("cv2.imread") as mock_imread:
            mock_imread.return_value = None  # OpenCV returns None for failed loads

            with pytest.raises(RuntimeError, match="Error loading image"):
                ImageUtils.load_image(test_image)

    def test_save_image_success(self, tmp_path: Path) -> None:
        """Test successful image saving."""
        output_path = tmp_path / "output.jpg"
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 128, 64]

        with patch("cv2.imwrite") as mock_imwrite:
            mock_imwrite.return_value = True

            ImageUtils.save_image(test_image, output_path, quality=90)

            mock_imwrite.assert_called_once()
            args, kwargs = mock_imwrite.call_args
            assert args[0] == str(output_path)
            assert np.array_equal(args[1], test_image)
            # Should have JPEG quality parameter
            assert args[2] == [1, 90]  # IMWRITE_JPEG_QUALITY = 1

    def test_save_image_png_compression(self, tmp_path: Path) -> None:
        """Test PNG image saving with compression."""
        output_path = tmp_path / "output.png"
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch("cv2.imwrite") as mock_imwrite:
            mock_imwrite.return_value = True

            ImageUtils.save_image(test_image, output_path, quality=70)

            # PNG compression should be calculated from quality
            args, kwargs = mock_imwrite.call_args
            # quality=70 -> compression = (100-70)//11 = 2
            assert args[2] == [16, 2]  # IMWRITE_PNG_COMPRESSION = 16

    def test_save_image_webp(self, tmp_path: Path) -> None:
        """Test WebP image saving."""
        output_path = tmp_path / "output.webp"
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch("cv2.imwrite") as mock_imwrite:
            mock_imwrite.return_value = True

            ImageUtils.save_image(test_image, output_path, quality=85)

            args, kwargs = mock_imwrite.call_args
            assert args[2] == [64, 85]  # IMWRITE_WEBP_QUALITY = 64

    def test_save_image_empty(self) -> None:
        """Test saving empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Cannot save empty image"):
            ImageUtils.save_image(empty_image, Path("output.jpg"))

    def test_save_image_invalid_type(self) -> None:
        """Test saving non-numpy array."""
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            ImageUtils.save_image("not an array", Path("output.jpg"))

    def test_save_image_opencv_fails(self, tmp_path: Path) -> None:
        """Test image saving when OpenCV fails."""
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        output_path = tmp_path / "output.jpg"

        with patch("cv2.imwrite") as mock_imwrite:
            mock_imwrite.return_value = False

            with pytest.raises(RuntimeError, match="Failed to save image"):
                ImageUtils.save_image(test_image, output_path)

    def test_resize_image_both_dimensions(self) -> None:
        """Test image resizing with both width and height."""
        original = np.zeros((100, 150, 3), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            resized = np.zeros((200, 300, 3), dtype=np.uint8)
            mock_resize.return_value = resized

            ImageUtils.resize_image(
                original, width=300, height=200, maintain_aspect_ratio=False
            )

            mock_resize.assert_called_once()
            args, kwargs = mock_resize.call_args
            assert args[1] == (300, 200)  # (width, height)

    def test_resize_image_maintain_aspect_ratio(self) -> None:
        """Test image resizing maintaining aspect ratio."""
        # 100x150 image (aspect ratio 2:3)
        original = np.zeros((150, 100, 3), dtype=np.uint8)  # height, width

        with patch("cv2.resize") as mock_resize:
            resized = np.zeros((300, 200, 3), dtype=np.uint8)
            mock_resize.return_value = resized

            # Request 200x300 but maintain aspect ratio
            ImageUtils.resize_image(
                original, width=200, height=300, maintain_aspect_ratio=True
            )

            # Should pick smaller scale factor to maintain aspect ratio
            # Scale for width: 200/100 = 2.0
            # Scale for height: 300/150 = 2.0
            # Both are equal, so final size should be 200x300
            mock_resize.assert_called_once()
            args, kwargs = mock_resize.call_args
            assert args[1] == (200, 300)

    def test_resize_image_width_only(self) -> None:
        """Test image resizing with width only."""
        # 100x150 image
        original = np.zeros((150, 100, 3), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            resized = np.zeros((300, 200, 3), dtype=np.uint8)
            mock_resize.return_value = resized

            ImageUtils.resize_image(original, width=200, maintain_aspect_ratio=True)

            # Scale = 200/100 = 2.0
            # New height = 150 * 2.0 = 300
            mock_resize.assert_called_once()
            args, kwargs = mock_resize.call_args
            assert args[1] == (200, 300)

    def test_resize_image_height_only(self) -> None:
        """Test image resizing with height only."""
        original = np.zeros((150, 100, 3), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            resized = np.zeros((200, 133, 3), dtype=np.uint8)
            mock_resize.return_value = resized

            ImageUtils.resize_image(original, height=200, maintain_aspect_ratio=True)

            # Scale = 200/150 = 1.333...
            # New width = 100 * 1.333 = 133.33 -> 133
            mock_resize.assert_called_once()
            args, kwargs = mock_resize.call_args
            assert args[1] == (133, 200)

    def test_resize_image_no_dimensions(self) -> None:
        """Test image resizing without specifying dimensions."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Must specify at least one dimension"):
            ImageUtils.resize_image(original)

    def test_resize_image_empty(self) -> None:
        """Test resizing empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Cannot resize empty image"):
            ImageUtils.resize_image(empty_image, width=100)

    def test_resize_image_invalid_interpolation(self) -> None:
        """Test resizing with invalid interpolation method."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid interpolation"):
            ImageUtils.resize_image(original, width=200, interpolation="INVALID")

    def test_crop_image_success(self) -> None:
        """Test successful image cropping."""
        # 100x100 image
        original = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = BoundingBox(x=20, y=30, width=40, height=30)

        result = ImageUtils.crop_image(original, bbox)

        # Should return a copy of the cropped region
        assert result.shape == (30, 40, 3)  # height, width, channels

    def test_crop_image_empty(self):
        """Test cropping empty image."""
        empty_image = np.array([])
        bbox = BoundingBox(x=0, y=0, width=10, height=10)

        with pytest.raises(ValueError, match="Cannot crop empty image"):
            ImageUtils.crop_image(empty_image, bbox)

    def test_crop_image_negative_coordinates(self):
        """Test cropping with negative coordinates."""
        # BoundingBox validation prevents negative coordinates
        # Negative x
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BoundingBox(x=-5, y=10, width=20, height=20)

        # Negative y
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BoundingBox(x=10, y=-5, width=20, height=20)

    def test_crop_image_exceeds_bounds(self):
        """Test cropping that exceeds image bounds."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)  # 100x100 image

        # Crop extends beyond image width
        bbox = BoundingBox(x=90, y=10, width=20, height=20)  # x+width = 110 > 100
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            ImageUtils.crop_image(original, bbox)

        # Crop extends beyond image height
        bbox = BoundingBox(x=10, y=90, width=20, height=20)  # y+height = 110 > 100
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            ImageUtils.crop_image(original, bbox)

    def test_calculate_image_hash_success(self):
        """Test successful image hash calculation."""
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # Test SHA256 (default)
        hash_result = ImageUtils.calculate_image_hash(test_image)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length

        # Test with specific hash type
        hash_result = ImageUtils.calculate_image_hash(test_image, hash_type="sha256")
        assert len(hash_result) == 64

    def test_calculate_image_hash_empty(self):
        """Test hash calculation with empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Cannot hash empty image"):
            ImageUtils.calculate_image_hash(empty_image)

    def test_calculate_image_hash_unsupported_type(self):
        """Test hash calculation with unsupported hash type."""
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported hash type"):
            ImageUtils.calculate_image_hash(test_image, hash_type="md5")

    def test_calculate_image_hash_consistency(self):
        """Test that identical images produce identical hashes."""
        test_image1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        test_image2 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        test_image3 = np.ones((50, 50, 3), dtype=np.uint8) * 129  # Different

        hash1 = ImageUtils.calculate_image_hash(test_image1)
        hash2 = ImageUtils.calculate_image_hash(test_image2)
        hash3 = ImageUtils.calculate_image_hash(test_image3)

        assert hash1 == hash2  # Identical images
        assert hash1 != hash3  # Different images

    def test_calculate_structural_similarity_identical(self):
        """Test SSIM calculation with identical images."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        with (
            patch.object(ImageUtils, "_compute_ssim_manual", return_value=1.0),
            patch("cv2.cvtColor") as mock_cvtcolor,
        ):
            gray_image = np.mean(test_image, axis=2).astype(np.uint8)
            mock_cvtcolor.return_value = gray_image

            ssim = ImageUtils.calculate_structural_similarity(test_image, test_image)
            assert ssim == 1.0  # Identical images should have SSIM = 1.0

    def test_calculate_structural_similarity_different_sizes(self):
        """Test SSIM calculation with different sized images."""
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image2 = np.ones((50, 50, 3), dtype=np.uint8) * 128

        with patch("cv2.cvtColor") as mock_cvtcolor:
            # Mock grayscale conversion
            gray1 = np.ones((100, 100), dtype=np.uint8) * 128
            gray2 = np.ones((50, 50), dtype=np.uint8) * 128
            mock_cvtcolor.side_effect = [gray1, gray2]

            with patch("cv2.resize") as mock_resize:
                # Mock resize to make images same size
                resized = np.ones((50, 50), dtype=np.uint8) * 128
                mock_resize.side_effect = [resized, gray2]

                with patch.object(
                    ImageUtils, "_compute_ssim_manual", return_value=0.95
                ):
                    ssim = ImageUtils.calculate_structural_similarity(image1, image2)
                    assert 0 <= ssim <= 1.0

    def test_calculate_structural_similarity_empty(self):
        """Test SSIM calculation with empty images."""
        empty_image1 = np.array([])
        empty_image2 = np.array([])

        with pytest.raises(ValueError, match="Cannot compare empty images"):
            ImageUtils.calculate_structural_similarity(empty_image1, empty_image2)

    def test_compute_ssim_manual(self):
        """Test manual SSIM computation."""
        # Create two similar images
        image1 = np.ones((50, 50), dtype=np.uint8) * 128
        image2 = np.ones((50, 50), dtype=np.uint8) * 130  # Slightly different

        with patch("cv2.GaussianBlur") as mock_blur:
            # Mock Gaussian blur to return the input (for simplicity)
            mock_blur.side_effect = lambda img, *args: img.astype(np.float64)

            ssim = ImageUtils._compute_ssim_manual(image1, image2)
            assert isinstance(ssim, float)
            assert 0 <= ssim <= 1.0

    def test_calculate_histogram_similarity_correlation(self):
        """Test histogram similarity using correlation method."""
        # Create two similar images
        image1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        image2 = np.ones((50, 50, 3), dtype=np.uint8) * 130

        with patch("cv2.cvtColor") as mock_cvtcolor:
            gray1 = np.ones((50, 50), dtype=np.uint8) * 128
            gray2 = np.ones((50, 50), dtype=np.uint8) * 130
            mock_cvtcolor.side_effect = [gray1, gray2]

            with patch("cv2.calcHist") as mock_calchist:
                # Mock histogram calculation
                hist1 = np.random.rand(256, 1).astype(np.float32)
                hist2 = np.random.rand(256, 1).astype(np.float32)
                mock_calchist.side_effect = [hist1, hist2]

                with (
                    patch("cv2.normalize"),
                    patch("cv2.compareHist") as mock_compare,
                ):
                    mock_compare.return_value = 0.8  # High correlation

                    similarity = ImageUtils.calculate_histogram_similarity(
                        image1, image2, method="correlation"
                    )

                    assert 0 <= similarity <= 1.0
                    mock_compare.assert_called_once_with(
                        hist1, hist2, 0
                    )  # HISTCMP_CORREL = 0

    def test_calculate_histogram_similarity_chi_square(self):
        """Test histogram similarity using chi-square method."""
        image1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        image2 = np.ones((50, 50, 3), dtype=np.uint8) * 130

        with patch("cv2.cvtColor") as mock_cvtcolor:
            gray1 = np.ones((50, 50), dtype=np.uint8) * 128
            gray2 = np.ones((50, 50), dtype=np.uint8) * 130
            mock_cvtcolor.side_effect = [gray1, gray2]

            with patch("cv2.calcHist") as mock_calchist:
                hist1 = np.random.rand(256, 1).astype(np.float32)
                hist2 = np.random.rand(256, 1).astype(np.float32)
                mock_calchist.side_effect = [hist1, hist2]

                with (
                    patch("cv2.normalize"),
                    patch("cv2.compareHist") as mock_compare,
                ):
                    mock_compare.return_value = 2.0  # Chi-square distance

                    similarity = ImageUtils.calculate_histogram_similarity(
                        image1, image2, method="chi_square"
                    )

                    # For chi-square, lower values are better, so result should be inverted
                    assert 0 <= similarity <= 1.0
                    assert similarity == 1.0 / (1.0 + 2.0)  # Expected inversion

    def test_calculate_histogram_similarity_invalid_method(self) -> None:
        """Test histogram similarity with invalid method."""
        image1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        image2 = np.ones((50, 50, 3), dtype=np.uint8) * 130

        with pytest.raises(ValueError, match="Invalid method"):
            ImageUtils.calculate_histogram_similarity(image1, image2, method="invalid")

    def test_find_template_matches_success(self) -> None:
        """Test successful template matching."""
        # 200x200 source image
        source = np.zeros((200, 200), dtype=np.uint8)
        source[50:100, 50:100] = 255  # White square

        # 50x50 template (matches the white square)
        template = np.ones((50, 50), dtype=np.uint8) * 255

        with patch("cv2.cvtColor") as mock_cvtcolor:
            # Mock grayscale conversion (images already grayscale)
            mock_cvtcolor.side_effect = [source, template]

            with patch("cv2.matchTemplate") as mock_match:
                # Mock template matching result
                result_map = np.zeros((151, 151), dtype=np.float32)  # 200-50+1 = 151
                result_map[50, 50] = 0.95  # High match at position (50, 50)
                result_map[100, 100] = 0.85  # Another match
                mock_match.return_value = result_map

                matches = ImageUtils.find_template_matches(
                    source, template, threshold=0.8, method="TM_CCOEFF_NORMED"
                )

                assert len(matches) == 2  # Should find 2 matches
                assert all(match["confidence"] >= 0.8 for match in matches)
                assert (
                    matches[0]["confidence"] >= matches[1]["confidence"]
                )  # Sorted by confidence

                # Check match structure
                first_match = matches[0]
                assert "bbox" in first_match
                assert "confidence" in first_match
                assert "center" in first_match
                assert "method" in first_match
                assert isinstance(first_match["bbox"], BoundingBox)

    def test_find_template_matches_no_matches(self) -> None:
        """Test template matching with no matches above threshold."""
        source = np.zeros((200, 200), dtype=np.uint8)
        template = np.ones((50, 50), dtype=np.uint8) * 255

        with patch("cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.side_effect = [source, template]

            with patch("cv2.matchTemplate") as mock_match:
                # All values below threshold
                result_map = np.ones((151, 151), dtype=np.float32) * 0.5
                mock_match.return_value = result_map

                matches = ImageUtils.find_template_matches(
                    source, template, threshold=0.8
                )

                assert len(matches) == 0

    def test_find_template_matches_sqdiff_method(self) -> None:
        """Test template matching with squared difference method."""
        source = np.zeros((100, 100), dtype=np.uint8)
        template = np.ones((25, 25), dtype=np.uint8) * 128

        with patch("cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.side_effect = [source, template]

            with patch("cv2.matchTemplate") as mock_match:
                # For SQDIFF, lower values are better matches
                result_map = np.ones((76, 76), dtype=np.float32) * 0.8
                result_map[25, 25] = 0.1  # Good match (low value)
                mock_match.return_value = result_map

                matches = ImageUtils.find_template_matches(
                    source, template, threshold=0.8, method="TM_SQDIFF_NORMED"
                )

                assert len(matches) == 1
                # For SQDIFF, confidence should be inverted
                assert matches[0]["confidence"] == pytest.approx(
                    1.0 - 0.1
                )  # Should be ~0.9

    def test_find_template_matches_empty_images(self) -> None:
        """Test template matching with empty images."""
        empty_source = np.array([])
        empty_template = np.array([])

        with pytest.raises(ValueError, match="Cannot match with empty images"):
            ImageUtils.find_template_matches(empty_source, empty_template)

    def test_find_template_matches_invalid_method(self) -> None:
        """Test template matching with invalid method."""
        source = np.zeros((100, 100), dtype=np.uint8)
        template = np.ones((25, 25), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid method"):
            ImageUtils.find_template_matches(source, template, method="INVALID")

    def test_find_template_matches_invalid_threshold(self) -> None:
        """Test template matching with invalid threshold."""
        source = np.zeros((100, 100), dtype=np.uint8)
        template = np.ones((25, 25), dtype=np.uint8)

        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            ImageUtils.find_template_matches(source, template, threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            ImageUtils.find_template_matches(source, template, threshold=-0.1)

    def test_validate_image_file_success(self, tmp_path: Path) -> None:
        """Test successful image file validation."""
        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake jpeg data" * 100)  # ~1.3KB

        with patch.object(ImageUtils, "load_image") as mock_load:
            # Mock successful image loading
            fake_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            mock_load.return_value = fake_image

            result = ImageUtils.validate_image_file(test_image)

            assert result.is_valid is True
            assert result.context["width"] == 640
            assert result.context["height"] == 480
            assert result.context["channels"] == 3
            assert result.context["megapixels"] == pytest.approx(0.31, rel=0.1)

    def test_validate_image_file_not_found(self):
        """Test image file validation with missing file."""
        result = ImageUtils.validate_image_file(Path("nonexistent.jpg"))

        assert result.is_valid is False
        assert any("not found" in error for error in result.errors)

    def test_validate_image_file_directory(self, tmp_path: Path) -> None:
        """Test image file validation with directory."""
        result = ImageUtils.validate_image_file(tmp_path)

        assert result.is_valid is False
        assert any("not a file" in error for error in result.errors)

    def test_validate_image_file_unusual_extension(self, tmp_path: Path) -> None:
        """Test image file validation with unusual extension."""
        weird_file = tmp_path / "image.xyz"
        weird_file.write_bytes(b"fake data")

        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.return_value = np.ones((100, 100, 3), dtype=np.uint8)

            result = ImageUtils.validate_image_file(weird_file)

            assert any(
                "Unusual image extension" in warning for warning in result.warnings
            )

    def test_validate_image_file_empty(self, tmp_path: Path) -> None:
        """Test image file validation with empty file."""
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")

        result = ImageUtils.validate_image_file(empty_file)

        assert result.is_valid is False
        assert any("empty" in error for error in result.errors)

    def test_validate_image_file_size_warnings(self, tmp_path: Path) -> None:
        """Test image file validation size warnings."""
        # Very small file
        tiny_file = tmp_path / "tiny.jpg"
        tiny_file.write_bytes(b"x" * 50)  # 50 bytes

        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.return_value = np.ones((10, 10, 3), dtype=np.uint8)

            result = ImageUtils.validate_image_file(tiny_file)

            assert any(
                "Very small image file" in warning for warning in result.warnings
            )

    def test_validate_image_file_dimension_warnings(self, tmp_path: Path) -> None:
        """Test image file validation dimension warnings."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake data" * 100)

        # Very small dimensions
        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.return_value = np.ones((8, 8, 3), dtype=np.uint8)

            result = ImageUtils.validate_image_file(test_file)

            assert any(
                "Very small image dimensions" in warning for warning in result.warnings
            )

        # Very large dimensions
        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.return_value = np.ones((10000, 10000, 3), dtype=np.uint8)

            result = ImageUtils.validate_image_file(test_file)

            assert any(
                "Very large image dimensions" in warning for warning in result.warnings
            )

    def test_validate_image_file_aspect_ratio_warning(self, tmp_path: Path) -> None:
        """Test image file validation aspect ratio warning."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake data" * 100)

        with patch.object(ImageUtils, "load_image") as mock_load:
            # Very wide image (unusual aspect ratio)
            mock_load.return_value = np.ones((100, 2000, 3), dtype=np.uint8)

            result = ImageUtils.validate_image_file(test_file)

            assert any("Unusual aspect ratio" in warning for warning in result.warnings)

    def test_validate_image_file_load_error(self, tmp_path: Path) -> None:
        """Test image file validation with load error."""
        test_file = tmp_path / "corrupted.jpg"
        test_file.write_bytes(b"corrupted data")

        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.side_effect = ValueError("Corrupted image")

            result = ImageUtils.validate_image_file(test_file)

            assert result.is_valid is False
            assert any("Invalid image file" in error for error in result.errors)

    def test_get_image_info_success(self, tmp_path: Path) -> None:
        """Test successful image info retrieval."""
        test_file = tmp_path / "info_test.jpg"
        test_file.write_bytes(b"fake image data" * 1000)  # ~15KB

        with patch.object(ImageUtils, "load_image") as mock_load:
            # Mock 1920x1080 RGB image
            mock_load.return_value = np.ones((1080, 1920, 3), dtype=np.uint8)

            info = ImageUtils.get_image_info(test_file)

            assert info["width"] == 1920
            assert info["height"] == 1080
            assert info["channels"] == 3
            assert info["dtype"] == "uint8"
            assert info["file_size"] > 0
            assert info["file_size_mb"] > 0
            assert info["megapixels"] == pytest.approx(2.07, rel=0.1)
            assert info["aspect_ratio"] == pytest.approx(1.78, rel=0.1)  # 16:9
            assert info["color_mode"] == "3-channel"

    def test_get_image_info_grayscale(self, tmp_path: Path) -> None:
        """Test image info for grayscale image."""
        test_file = tmp_path / "gray.jpg"
        test_file.write_bytes(b"gray data")

        with patch.object(ImageUtils, "load_image") as mock_load:
            mock_load.return_value = np.ones((480, 640), dtype=np.uint8)  # Grayscale

            info = ImageUtils.get_image_info(test_file)

            assert info["channels"] == 1
            assert info["color_mode"] == "grayscale"

    def test_get_image_info_file_not_found(self) -> None:
        """Test image info with missing file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            ImageUtils.get_image_info(Path("missing.jpg"))

    def test_create_color_mask_hsv(self) -> None:
        """Test color mask creation in HSV color space."""
        # RGB image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        lower_hsv = (100, 50, 50)
        upper_hsv = (130, 255, 255)

        with patch("cv2.cvtColor") as mock_cvtcolor:
            hsv_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
            mock_cvtcolor.return_value = hsv_image

            with patch("cv2.inRange") as mock_inrange:
                mask = np.zeros((100, 100), dtype=np.uint8)
                mask[25:75, 25:75] = 255  # White region in mask
                mock_inrange.return_value = mask

                result = ImageUtils.create_color_mask(
                    test_image, lower_hsv, upper_hsv, color_space="HSV"
                )

                mock_cvtcolor.assert_called_once_with(
                    test_image, 40
                )  # COLOR_BGR2HSV = 40
                mock_inrange.assert_called_once_with(hsv_image, lower_hsv, upper_hsv)
                assert result.shape == (100, 100)

    def test_create_color_mask_bgr(self) -> None:
        """Test color mask creation in BGR color space."""
        test_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        lower_bgr = (0, 0, 100)
        upper_bgr = (50, 50, 200)

        with patch("cv2.inRange") as mock_inrange:
            mask = np.zeros((50, 50), dtype=np.uint8)
            mock_inrange.return_value = mask

            ImageUtils.create_color_mask(
                test_image, lower_bgr, upper_bgr, color_space="BGR"
            )

            # No color conversion should happen for BGR
            mock_inrange.assert_called_once_with(test_image, lower_bgr, upper_bgr)

    def test_create_color_mask_empty_image(self) -> None:
        """Test color mask creation with empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Cannot create mask from empty image"):
            ImageUtils.create_color_mask(empty_image, (0, 0, 0), (255, 255, 255))

    def test_create_color_mask_invalid_color_space(self) -> None:
        """Test color mask creation with invalid color space."""
        test_image = np.ones((50, 50, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid color_space"):
            ImageUtils.create_color_mask(
                test_image, (0, 0, 0), (255, 255, 255), color_space="INVALID"
            )
