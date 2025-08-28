"""Minimal backend factory with lightweight adapters for a unified API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mivideoeditor.ml.backends.base import (
    BaseEvaluator,
    BasePredictor,
    BaseTrainer,
    StandardPrediction,
)
from mivideoeditor.ml.engine.evaluator import Evaluator, HFEvaluator
from mivideoeditor.ml.engine.predictor import HFPredictor, Predictor
from mivideoeditor.ml.engine.predictor import (
    TorchSegmentationPredictor as _SegPred,
)
from mivideoeditor.ml.engine.trainer import HFTrainer, Trainer


@dataclass
class _TorchPredictorAdapter(BasePredictor):
    impl: Any  # mivideoeditor.ml.engine.predictor.Predictor

    def predict_numpy(self, image_bgr: np.ndarray) -> StandardPrediction:
        out = self.impl.predict_numpy(image_bgr)
        return StandardPrediction(boxes=out.boxes, scores=out.scores, labels=out.labels)


def _make_torch_predictor(task: str, **kwargs: Any) -> BasePredictor:
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    if task == "instance_segmentation":
        impl = _SegPred(**kwargs)
    else:
        impl = Predictor(**kwargs)
    if checkpoint_path:
        impl.load_checkpoint(Path(checkpoint_path))
    return _TorchPredictorAdapter(impl=impl)


def _make_torch_trainer(**kwargs: Any) -> BaseTrainer:
    return Trainer(**kwargs)


def _make_torch_evaluator(**kwargs: Any) -> BaseEvaluator:
    return Evaluator(**kwargs)


@dataclass
class _HFPredictorAdapter(BasePredictor):
    impl: Any  # mivideoeditor.ml.engine.hf_predictor.HFPredictor

    def predict_numpy(self, image_bgr: np.ndarray) -> StandardPrediction:
        out = self.impl.predict_numpy(image_bgr)
        return StandardPrediction(boxes=out.boxes, scores=out.scores, labels=out.labels)


def _make_hf_predictor(**kwargs: Any) -> BasePredictor:
    impl = HFPredictor(**kwargs)
    return _HFPredictorAdapter(impl=impl)


def _make_hf_trainer(**kwargs: Any) -> BaseTrainer:
    return HFTrainer(**kwargs)


def _make_hf_evaluator(**kwargs: Any) -> BaseEvaluator:
    return HFEvaluator(**kwargs)


def build_predictor(backend: str, task: str, **kwargs: Any) -> BasePredictor:
    """Return a predictor for the given backend and task (no DI registry)."""
    backend = (backend or "torch").lower()
    if backend == "torch":
        return _make_torch_predictor(task=task, **kwargs)
    if backend == "hf":
        return _make_hf_predictor(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


def build_trainer(backend: str, task: str, **kwargs: Any) -> BaseTrainer:
    """Return a trainer for the given backend and task."""
    backend = (backend or "torch").lower()
    if backend == "torch":
        return _make_torch_trainer(**kwargs)
    if backend == "hf":
        return _make_hf_trainer(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


def build_evaluator(backend: str, task: str, **kwargs: Any) -> BaseEvaluator:
    """Return an evaluator for the given backend and task."""
    backend = (backend or "torch").lower()
    if backend == "torch":
        return _make_torch_evaluator(**kwargs)
    if backend == "hf":
        return _make_hf_evaluator(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


# Backward-compat aliases (if any code imported previous names)
get_predictor = build_predictor
get_trainer = build_trainer
get_evaluator = build_evaluator
