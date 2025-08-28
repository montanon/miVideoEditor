"""Lightweight dependency-injection registry for ML backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Key = tuple[str, str]
Provider = Callable[..., Any]

_TRAINERS: dict[Key, Provider] = {}
_EVALUATORS: dict[Key, Provider] = {}
_PREDICTORS: dict[Key, Provider] = {}


def _norm(backend: str, task: str) -> Key:
    """Normalize backend and task names."""
    return (backend.lower(), task.lower())


def register_trainer(backend: str, task: str, provider: Provider) -> None:
    """Register a trainer for the given backend and task."""
    _TRAINERS[_norm(backend, task)] = provider


def register_evaluator(backend: str, task: str, provider: Provider) -> None:
    """Register an evaluator for the given backend and task."""
    _EVALUATORS[_norm(backend, task)] = provider


def register_predictor(backend: str, task: str, provider: Provider) -> None:
    """Register a predictor for the given backend and task."""
    _PREDICTORS[_norm(backend, task)] = provider


def get_trainer_provider(backend: str, task: str) -> Provider | None:
    """Get a trainer provider for the given backend and task."""
    return _TRAINERS.get(_norm(backend, task))


def get_evaluator_provider(backend: str, task: str) -> Provider | None:
    """Get an evaluator provider for the given backend and task."""
    return _EVALUATORS.get(_norm(backend, task))


def get_predictor_provider(backend: str, task: str) -> Provider | None:
    """Get a predictor provider for the given backend and task."""
    return _PREDICTORS.get(_norm(backend, task))
