"""Lazy-loading facade for training components.

This avoids import-time circular dependencies between loss modules and the
training package by deferring heavy submodule imports until the symbols are
actually needed. External callers can continue to use
``from metta.rl.training import Evaluator`` and similar statements without
change.
"""

import importlib
from typing import TYPE_CHECKING, Any

# Type imports only happen during type checking, not at runtime
if TYPE_CHECKING:
    from metta.rl.training.checkpointer import Checkpointer, CheckpointerConfig
    from metta.rl.training.component import TrainerCallback, TrainerComponent
    from metta.rl.training.component_context import ComponentContext, TrainerState, TrainingEnvWindow
    from metta.rl.training.context_checkpointer import ContextCheckpointer
    from metta.rl.training.core import CoreTrainingLoop, RolloutResult
    from metta.rl.training.distributed_helper import DistributedHelper
    from metta.rl.training.evaluator import Evaluator, EvaluatorConfig, NoOpEvaluator
    from metta.rl.training.experience import Experience
    from metta.rl.training.gradient_reporter import GradientReporter, GradientReporterConfig
    from metta.rl.training.heartbeat import Heartbeat, HeartbeatConfig
    from metta.rl.training.monitor import Monitor
    from metta.rl.training.progress_logger import ProgressLogger
    from metta.rl.training.scheduler import HyperparameterSchedulerConfig, Scheduler, SchedulerConfig
    from metta.rl.training.stats_reporter import (
        NoOpStatsReporter,
        StatsReporter,
        StatsReporterConfig,
        StatsReporterState,
    )
    from metta.rl.training.torch_profiler import TorchProfiler
    from metta.rl.training.training_environment import (
        GameRules,
        TrainingEnvironment,
        TrainingEnvironmentConfig,
        VectorizedTrainingEnvironment,
    )
    from metta.rl.training.uploader import Uploader, UploaderConfig
    from metta.rl.training.wandb_aborter import WandbAborter, WandbAborterConfig
    from metta.rl.training.wandb_logger import WandbLogger

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Checkpointer": ("metta.rl.training.checkpointer", "Checkpointer"),
    "CheckpointerConfig": ("metta.rl.training.checkpointer", "CheckpointerConfig"),
    "ComponentContext": ("metta.rl.training.component_context", "ComponentContext"),
    "ContextCheckpointer": ("metta.rl.training.context_checkpointer", "ContextCheckpointer"),
    "CoreTrainingLoop": ("metta.rl.training.core", "CoreTrainingLoop"),
    "DistributedHelper": ("metta.rl.training.distributed_helper", "DistributedHelper"),
    "GameRules": ("metta.rl.training.training_environment", "GameRules"),
    "Evaluator": ("metta.rl.training.evaluator", "Evaluator"),
    "EvaluatorConfig": ("metta.rl.training.evaluator", "EvaluatorConfig"),
    "Experience": ("metta.rl.training.experience", "Experience"),
    "GradientReporter": ("metta.rl.training.gradient_reporter", "GradientReporter"),
    "GradientReporterConfig": ("metta.rl.training.gradient_reporter", "GradientReporterConfig"),
    "Heartbeat": ("metta.rl.training.heartbeat", "Heartbeat"),
    "HeartbeatConfig": ("metta.rl.training.heartbeat", "HeartbeatConfig"),
    "HyperparameterSchedulerConfig": ("metta.rl.training.scheduler", "HyperparameterSchedulerConfig"),
    "Monitor": ("metta.rl.training.monitor", "Monitor"),
    "NoOpEvaluator": ("metta.rl.training.evaluator", "NoOpEvaluator"),
    "NoOpStatsReporter": ("metta.rl.training.stats_reporter", "NoOpStatsReporter"),
    "ProgressLogger": ("metta.rl.training.progress_logger", "ProgressLogger"),
    "RolloutResult": ("metta.rl.training.core", "RolloutResult"),
    "Scheduler": ("metta.rl.training.scheduler", "Scheduler"),
    "SchedulerConfig": ("metta.rl.training.scheduler", "SchedulerConfig"),
    "StatsReporter": ("metta.rl.training.stats_reporter", "StatsReporter"),
    "StatsReporterConfig": ("metta.rl.training.stats_reporter", "StatsReporterConfig"),
    "StatsReporterState": ("metta.rl.training.stats_reporter", "StatsReporterState"),
    "TorchProfiler": ("metta.rl.training.torch_profiler", "TorchProfiler"),
    "TrainerCallback": ("metta.rl.training.component", "TrainerCallback"),
    "TrainerComponent": ("metta.rl.training.component", "TrainerComponent"),
    "TrainerState": ("metta.rl.training.component_context", "TrainerState"),
    "TrainingEnvironment": ("metta.rl.training.training_environment", "TrainingEnvironment"),
    "TrainingEnvironmentConfig": ("metta.rl.training.training_environment", "TrainingEnvironmentConfig"),
    "TrainingEnvWindow": ("metta.rl.training.component_context", "TrainingEnvWindow"),
    "Uploader": ("metta.rl.training.uploader", "Uploader"),
    "UploaderConfig": ("metta.rl.training.uploader", "UploaderConfig"),
    "VectorizedTrainingEnvironment": ("metta.rl.training.training_environment", "VectorizedTrainingEnvironment"),
    "WandbAborter": ("metta.rl.training.wandb_aborter", "WandbAborter"),
    "WandbAborterConfig": ("metta.rl.training.wandb_aborter", "WandbAborterConfig"),
    "WandbLogger": ("metta.rl.training.wandb_logger", "WandbLogger"),
    "UpdateEpochAutoTuner": ("metta.rl.training.update_epochs_tuner", "UpdateEpochAutoTuner"),
}

# Explicitly define __all__ to help type checkers
__all__ = [
    "Checkpointer",
    "CheckpointerConfig",
    "ComponentContext",
    "ContextCheckpointer",
    "CoreTrainingLoop",
    "DistributedHelper",
    "GameRules",
    "Evaluator",
    "EvaluatorConfig",
    "Experience",
    "GradientReporter",
    "GradientReporterConfig",
    "Heartbeat",
    "HeartbeatConfig",
    "HyperparameterSchedulerConfig",
    "Monitor",
    "NoOpEvaluator",
    "NoOpStatsReporter",
    "ProgressLogger",
    "RolloutResult",
    "Scheduler",
    "SchedulerConfig",
    "StatsReporter",
    "StatsReporterConfig",
    "StatsReporterState",
    "TorchProfiler",
    "TrainerCallback",
    "TrainerComponent",
    "TrainerState",
    "TrainingEnvironment",
    "TrainingEnvironmentConfig",
    "TrainingEnvWindow",
    "Uploader",
    "UploaderConfig",
    "VectorizedTrainingEnvironment",
    "WandbAborter",
    "WandbAborterConfig",
    "WandbLogger",
    "UpdateEpochAutoTuner",
]


def __getattr__(name: str) -> Any:
    """Dynamically import training submodules on attribute access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'metta.rl.training' has no attribute '{name}'")

    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    attr = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
