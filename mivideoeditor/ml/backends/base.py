"""Thin backend interfaces and shared prediction structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class StandardPrediction:
    """Backend-agnostic prediction result."""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)


class BasePredictor(Protocol):
    """Base predictor protocol."""

    def predict_numpy(self, image_bgr: np.ndarray) -> StandardPrediction:
        """Predict on an OpenCV-style BGR numpy image."""
        ...


class BaseTrainer(Protocol):
    """Base trainer protocol."""

    def fit(self) -> Any:
        """Train the model."""
        ...


class BaseEvaluator(Protocol):
    """Base evaluator protocol."""

    def evaluate(self) -> Any:
        """Evaluate the model."""
        ...
