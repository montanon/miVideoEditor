"""Backend factory and lightweight adapters for unified API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mivideoeditor.ml.backends.base import (
    BaseEvaluator,
    BasePredictor,
    BaseTrainer,
    StandardPrediction,
)
from mivideoeditor.ml.backends.registry import (
    get_evaluator_provider,
    get_predictor_provider,
    get_trainer_provider,
)
from mivideoeditor.ml.engine.evaluator import Evaluator, HFEvaluator
from mivideoeditor.ml.engine.predictor import HFPredictor, Predictor
from mivideoeditor.ml.engine.trainer import HFTrainer, Trainer


@dataclass
class _TorchPredictorAdapter(BasePredictor):
    impl: Any  # mivideoeditor.ml.engine.predictor.Predictor

    def predict_numpy(self, image_bgr: np.ndarray) -> StandardPrediction:
        out = self.impl.predict_numpy(image_bgr)
        return StandardPrediction(boxes=out.boxes, scores=out.scores, labels=out.labels)


def _make_torch_predictor(**kwargs: Any) -> BasePredictor:
    impl = Predictor(**kwargs)
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


def get_predictor(backend: str, task: str, **kwargs: Any) -> BasePredictor:
    """Return a predictor for the given backend and task."""
    backend = (backend or "torch").lower()
    provider = get_predictor_provider(backend, task)
    if provider is not None:
        impl = provider(**kwargs)
        # If provider already returns StandardPrediction-capable impl, pass-through
        if hasattr(impl, "predict_numpy"):
            return impl  # type: ignore[return-value]
        msg = "Predictor provider must return an object with predict_numpy"
        raise TypeError(msg)
    if backend == "torch":
        return _make_torch_predictor(**kwargs)
    if backend == "hf":
        return _make_hf_predictor(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


def get_trainer(backend: str, task: str, **kwargs: Any) -> BaseTrainer:
    """Return a trainer for the given backend and task."""
    backend = (backend or "torch").lower()
    provider = get_trainer_provider(backend, task)
    if provider is not None:
        return provider(**kwargs)
    if backend == "torch":
        return _make_torch_trainer(**kwargs)
    if backend == "hf":
        return _make_hf_trainer(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


def get_evaluator(backend: str, task: str, **kwargs: Any) -> BaseEvaluator:
    """Return an evaluator for the given backend and task."""
    backend = (backend or "torch").lower()
    provider = get_evaluator_provider(backend, task)
    if provider is not None:
        return provider(**kwargs)
    if backend == "torch":
        return _make_torch_evaluator(**kwargs)
    if backend == "hf":
        return _make_hf_evaluator(**kwargs)
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)
