"""Machine Learning submodule.

Provides decoupled training, evaluation, and prediction workflows and an
adapter to plug trained models into the existing detection pipeline.
"""

from mivideoeditor.ml.api import (
    MLEvaluator,
    MLPredictor,
    MLTrainer,
    Prediction,
)
from mivideoeditor.ml.backends.factory import (
    build_evaluator,
    build_predictor,
    build_trainer,
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
    # Unified interfaces
    "MLPredictor",
    "Prediction",
    "MLTrainer",
    "MLEvaluator",
    # HF Configs
    "HuggingFaceModelConfig",
    "HuggingFacePredictConfig",
    "HFTrainingConfig",
    # Backends factory
    "build_trainer",
    "build_evaluator",
    "build_predictor",
]
