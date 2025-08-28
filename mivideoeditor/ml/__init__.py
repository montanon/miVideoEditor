"""Machine Learning submodule.

Provides decoupled training, evaluation, and prediction workflows and an
adapter to plug trained models into the existing detection pipeline.
"""

from mivideoeditor.ml.backends.factory import (
    get_evaluator,
    get_predictor,
    get_trainer,
)
from mivideoeditor.ml.config import (
    DataConfig,
    EvalConfig,
    HFTrainingConfig,
    HuggingFaceModelConfig,
    HuggingFacePredictConfig,
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
    # HF Configs
    "HuggingFaceModelConfig",
    "HuggingFacePredictConfig",
    "HFTrainingConfig",
    # Backends factory
    "get_trainer",
    "get_evaluator",
    "get_predictor",
]
