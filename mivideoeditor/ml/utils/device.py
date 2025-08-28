"""Device selection utilities for CUDA/MPS/CPU."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Select an appropriate torch.device."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS may not exist on all builds
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
