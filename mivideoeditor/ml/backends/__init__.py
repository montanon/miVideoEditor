"""Backend implementations for ML pipeline."""

from mivideoeditor.ml.backends.base import (
    BaseEvaluator,
    BasePredictor,
    BaseTrainer,
    StandardPrediction,
)
from mivideoeditor.ml.backends.factory import (
    get_evaluator,
    get_predictor,
    get_trainer,
)

__all__ = [
    "BaseEvaluator",
    "BasePredictor",
    "BaseTrainer",
    "StandardPrediction",
    "get_evaluator",
    "get_predictor",
    "get_trainer",
]
