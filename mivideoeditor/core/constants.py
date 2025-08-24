"""System-wide constants for video privacy editor."""

# Supported area types
SUPPORTED_AREA_TYPES = {
    "chatgpt": "ChatGPT conversation interface",
    "atuin": "Atuin terminal history search",
    "terminal": "Generic terminal content",
    "sensitive_text": "Generic sensitive text content",
    "custom": "User-defined sensitive area",
}

# Blur filter types
BLUR_FILTER_TYPES = {
    "gaussian": "Gaussian blur filter",
    "pixelate": "Pixelation/mosaic filter",
    "noise": "Random noise overlay",
    "composite": "Combined blur effects",
}

# Interpolation modes
INTERPOLATION_MODES = ["linear", "smooth", "none"]
