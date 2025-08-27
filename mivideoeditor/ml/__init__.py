"""Machine Learning submodule.

Provides decoupled training, evaluation, and prediction workflows and an
adapter to plug trained models into the existing detection pipeline.
"""

from mivideoeditor.ml.config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    PipelineConfig,
    PredictConfig,
    TrainConfig,
)
from mivideoeditor.ml.engine.evaluator import Evaluator
from mivideoeditor.ml.engine.predictor import Predictor
from mivideoeditor.ml.engine.trainer import Trainer

__all__ = [
    # Configs
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "PredictConfig",
    "PipelineConfig",
    # Engines
    "Trainer",
    "Evaluator",
    "Predictor",
]
