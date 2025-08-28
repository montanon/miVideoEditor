"""Unified ML interfaces (SOLID-friendly) and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Prediction:
    """Backend-agnostic prediction result."""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)
    masks: np.ndarray | None = None  # (N, H, W) optional for segmentation


class MLPredictor(Protocol):
    """Minimal predictor interface consumed by MLDetector."""

    def predict_numpy(self, image_bgr: np.ndarray) -> Prediction:
        """Predict the model."""
        ...


@dataclass
class TrainArtifacts:
    """TrainArtifacts."""

    checkpoint_path: str | None
    best_metric: float | None
    best_epoch: int | None


@dataclass
class EvalReport:
    """EvalReport."""

    summary: dict
    per_class: dict


class MLTrainer(Protocol):
    """MLTrainer."""

    def fit(self) -> TrainArtifacts:
        """Fit the model."""
        ...


class MLEvaluator(Protocol):
    """MLEvaluator."""

    def evaluate(self) -> EvalReport:
        """Evaluate the model."""
        ...
