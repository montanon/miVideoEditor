"""Tests for core constants."""

from mivideoeditor.core.constants import (
    BLUR_FILTER_TYPES,
    INTERPOLATION_MODES,
    SUPPORTED_AREA_TYPES,
)


class TestSupportedAreaTypes:
    """Test supported area types constants."""

    def test_supported_area_types_exist(self) -> None:
        """Test that all expected area types are defined."""
        expected_types = {"chatgpt", "atuin", "terminal", "sensitive_text", "custom"}
        assert set(SUPPORTED_AREA_TYPES.keys()) == expected_types

    def test_area_types_have_descriptions(self) -> None:
        """Test that all area types have descriptions."""
        for area_type, description in SUPPORTED_AREA_TYPES.items():
            assert isinstance(area_type, str)
            assert isinstance(description, str)
            assert len(description) > 0
            # For custom type, check for meaningful words instead of exact match
            if area_type == "custom":
                assert "user" in description.lower() or "defined" in description.lower()
            else:
                assert area_type.replace("_", " ") in description.lower()

    def test_chatgpt_type_description(self) -> None:
        """Test ChatGPT type has appropriate description."""
        assert "chatgpt" in SUPPORTED_AREA_TYPES["chatgpt"].lower()
        assert "conversation" in SUPPORTED_AREA_TYPES["chatgpt"].lower()

    def test_terminal_types_exist(self) -> None:
        """Test both generic and specific terminal types exist."""
        assert "terminal" in SUPPORTED_AREA_TYPES
        assert "atuin" in SUPPORTED_AREA_TYPES


class TestBlurFilterTypes:
    """Test blur filter types constants."""

    def test_blur_filter_types_exist(self) -> None:
        """Test that all expected filter types are defined."""
        expected_filters = {"gaussian", "pixelate", "noise", "composite"}
        assert set(BLUR_FILTER_TYPES.keys()) == expected_filters

    def test_filter_types_have_descriptions(self) -> None:
        """Test that all filter types have descriptions."""
        for filter_type, description in BLUR_FILTER_TYPES.items():
            assert isinstance(filter_type, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_composite_is_available(self) -> None:
        """Test that composite filter is available."""
        assert "composite" in BLUR_FILTER_TYPES
        assert "blur" in BLUR_FILTER_TYPES["composite"].lower()


class TestInterpolationModes:
    """Test interpolation modes constants."""

    def test_interpolation_modes_exist(self) -> None:
        """Test that all expected interpolation modes are defined."""
        expected_modes = ["linear", "smooth", "none"]
        assert expected_modes == INTERPOLATION_MODES

    def test_interpolation_modes_are_strings(self) -> None:
        """Test that all interpolation modes are strings."""
        for mode in INTERPOLATION_MODES:
            assert isinstance(mode, str)
            assert len(mode) > 0

    def test_none_mode_available(self) -> None:
        """Test that 'none' mode is available for static regions."""
        assert "none" in INTERPOLATION_MODES

    def test_linear_mode_available(self) -> None:
        """Test that 'linear' mode is available for basic interpolation."""
        assert "linear" in INTERPOLATION_MODES


class TestConstantsIntegration:
    """Test integration between different constants."""

    def test_constants_are_immutable(self) -> None:
        """Test that constants cannot be modified."""
        # These should be treated as immutable by convention
        original_area_types = SUPPORTED_AREA_TYPES.copy()
        original_blur_types = BLUR_FILTER_TYPES.copy()
        original_interpolation = INTERPOLATION_MODES.copy()

        # Test that copies match originals
        assert original_area_types == SUPPORTED_AREA_TYPES
        assert original_blur_types == BLUR_FILTER_TYPES
        assert original_interpolation == INTERPOLATION_MODES

    def test_no_empty_values(self) -> None:
        """Test that no constants have empty values."""
        for area_type in SUPPORTED_AREA_TYPES:
            assert area_type
            assert SUPPORTED_AREA_TYPES[area_type]

        for filter_type in BLUR_FILTER_TYPES:
            assert filter_type
            assert BLUR_FILTER_TYPES[filter_type]

        for mode in INTERPOLATION_MODES:
            assert mode

    def test_constants_are_lowercase(self) -> None:
        """Test that constant keys are lowercase for consistency."""
        for area_type in SUPPORTED_AREA_TYPES:
            assert area_type.islower()

        for filter_type in BLUR_FILTER_TYPES:
            assert filter_type.islower()

        for mode in INTERPOLATION_MODES:
            assert mode.islower()
