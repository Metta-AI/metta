"""Public re-exports for the training package without eager circular imports."""

from __future__ import annotations

import importlib
from typing import Any

from . import training_environment
from .component import TrainerComponent
from .context import TrainerContext
from .distributed_helper import DistributedHelper
from .evaluator import Evaluator, EvaluatorConfig, NoOpEvaluator
from .gradient_stats import GradientStatsComponent, GradientStatsConfig
from .heartbeat import HeartbeatConfig, HeartbeatWriter
from .hyperparameter import HyperparameterComponent, HyperparameterConfig
from .monitoring_component import MonitoringComponent
from .policy_checkpointer import PolicyCheckpointer, PolicyCheckpointerConfig
from .policy_uploader import PolicyUploader, PolicyUploaderConfig
from .stats_reporter import NoOpStatsReporter, StatsConfig, StatsReporter, StatsState
from .torch_profiler_component import TorchProfilerComponent
from .trainer_checkpointer import TrainerCheckpointer, TrainerCheckpointerConfig
from .wandb_abort import WandbAbortComponent

__all__ = [
    "CoreTrainingLoop",
    "RolloutResult",
    "DistributedHelper",
    "PolicyCheckpointer",
    "PolicyCheckpointerConfig",
    "PolicyUploader",
    "PolicyUploaderConfig",
    "TrainerCheckpointer",
    "TrainerCheckpointerConfig",
    "TrainerContext",
    "WandbAbortComponent",
    "Evaluator",
    "EvaluatorConfig",
    "NoOpEvaluator",
    "StatsReporter",
    "StatsConfig",
    "StatsState",
    "NoOpStatsReporter",
    "TrainerComponent",
    "MonitoringComponent",
    "TorchProfilerComponent",
    "HeartbeatWriter",
    "HeartbeatConfig",
    "HyperparameterComponent",
    "HyperparameterConfig",
    "GradientStatsComponent",
    "GradientStatsConfig",
    "training_environment",
]

_LAZY_EXPORTS = {
    "CoreTrainingLoop": ("metta.rl.training.core", "CoreTrainingLoop"),
    "RolloutResult": ("metta.rl.training.core", "RolloutResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'metta.rl.training' has no attribute '{name}'")
