"""Public re-exports for the training package without eager circular imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import training_environment
from .component import TrainerComponent
from .context import TrainerContext
from .distributed_helper import DistributedHelper

if TYPE_CHECKING:  # pragma: no cover - only active during static analysis
    from metta.agent.policy import Policy as Policy
    from metta.agent.policy import PolicyArchitecture as PolicyArchitecture

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
else:  # pragma: no cover - runtime avoids circular import
    Policy = Any  # type: ignore[assignment]
    PolicyArchitecture = Any  # type: ignore[assignment]

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
    "Policy",
    "PolicyArchitecture",
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
    "Evaluator": ("metta.rl.training.evaluator", "Evaluator"),
    "EvaluatorConfig": ("metta.rl.training.evaluator", "EvaluatorConfig"),
    "NoOpEvaluator": ("metta.rl.training.evaluator", "NoOpEvaluator"),
    "GradientStatsComponent": (
        "metta.rl.training.gradient_stats",
        "GradientStatsComponent",
    ),
    "GradientStatsConfig": (
        "metta.rl.training.gradient_stats",
        "GradientStatsConfig",
    ),
    "HeartbeatConfig": ("metta.rl.training.heartbeat", "HeartbeatConfig"),
    "HeartbeatWriter": ("metta.rl.training.heartbeat", "HeartbeatWriter"),
    "HyperparameterComponent": (
        "metta.rl.training.hyperparameter",
        "HyperparameterComponent",
    ),
    "HyperparameterConfig": (
        "metta.rl.training.hyperparameter",
        "HyperparameterConfig",
    ),
    "MonitoringComponent": (
        "metta.rl.training.monitoring_component",
        "MonitoringComponent",
    ),
    "PolicyCheckpointer": (
        "metta.rl.training.policy_checkpointer",
        "PolicyCheckpointer",
    ),
    "PolicyCheckpointerConfig": (
        "metta.rl.training.policy_checkpointer",
        "PolicyCheckpointerConfig",
    ),
    "PolicyUploader": ("metta.rl.training.policy_uploader", "PolicyUploader"),
    "PolicyUploaderConfig": (
        "metta.rl.training.policy_uploader",
        "PolicyUploaderConfig",
    ),
    "StatsReporter": ("metta.rl.training.stats_reporter", "StatsReporter"),
    "StatsConfig": ("metta.rl.training.stats_reporter", "StatsConfig"),
    "StatsState": ("metta.rl.training.stats_reporter", "StatsState"),
    "NoOpStatsReporter": ("metta.rl.training.stats_reporter", "NoOpStatsReporter"),
    "TorchProfilerComponent": (
        "metta.rl.training.torch_profiler_component",
        "TorchProfilerComponent",
    ),
    "TrainerCheckpointer": (
        "metta.rl.training.trainer_checkpointer",
        "TrainerCheckpointer",
    ),
    "TrainerCheckpointerConfig": (
        "metta.rl.training.trainer_checkpointer",
        "TrainerCheckpointerConfig",
    ),
    "WandbAbortComponent": (
        "metta.rl.training.wandb_abort",
        "WandbAbortComponent",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'metta.rl.training' has no attribute '{name}'")
