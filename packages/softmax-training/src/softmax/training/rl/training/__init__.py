"""Lazy-loading facade for training components.

This avoids import-time circular dependencies between loss modules and the
training package by deferring heavy submodule imports until the symbols are
actually needed. External callers can continue to use
``from softmax.training.rl.training import Evaluator`` and similar statements without
change.
"""

import importlib
from typing import TYPE_CHECKING, Any

# Type imports only happen during type checking, not at runtime
if TYPE_CHECKING:
    from softmax.training.rl.training.checkpointer import Checkpointer, CheckpointerConfig
    from softmax.training.rl.training.component import TrainerCallback, TrainerComponent
    from softmax.training.rl.training.component_context import ComponentContext, TrainerState, TrainingEnvWindow
    from softmax.training.rl.training.context_checkpointer import ContextCheckpointer, ContextCheckpointerConfig
    from softmax.training.rl.training.core import CoreTrainingLoop, RolloutResult
    from softmax.training.rl.training.distributed_helper import DistributedHelper
    from softmax.training.rl.training.evaluator import Evaluator, EvaluatorConfig, NoOpEvaluator
    from softmax.training.rl.training.experience import Experience
    from softmax.training.rl.training.gradient_reporter import GradientReporter, GradientReporterConfig
    from softmax.training.rl.training.heartbeat import Heartbeat, HeartbeatConfig
    from softmax.training.rl.training.monitor import Monitor
    from softmax.training.rl.training.progress_logger import ProgressLogger
    from softmax.training.rl.training.scheduler import HyperparameterSchedulerConfig, Scheduler, SchedulerConfig
    from softmax.training.rl.training.stats_reporter import (
        NoOpStatsReporter,
        StatsReporter,
        StatsReporterConfig,
        StatsReporterState,
    )
    from softmax.training.rl.training.torch_profiler import TorchProfiler
    from softmax.training.rl.training.training_environment import (
        EnvironmentMetaData,
        TrainingEnvironment,
        TrainingEnvironmentConfig,
        VectorizedTrainingEnvironment,
    )
    from softmax.training.rl.training.uploader import Uploader, UploaderConfig
    from softmax.training.rl.training.wandb_aborter import WandbAborter, WandbAborterConfig
    from softmax.training.rl.training.wandb_logger import WandbLogger

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Checkpointer": ("softmax.training.rl.training.checkpointer", "Checkpointer"),
    "CheckpointerConfig": ("softmax.training.rl.training.checkpointer", "CheckpointerConfig"),
    "ComponentContext": ("softmax.training.rl.training.component_context", "ComponentContext"),
    "ContextCheckpointer": ("softmax.training.rl.training.context_checkpointer", "ContextCheckpointer"),
    "ContextCheckpointerConfig": ("softmax.training.rl.training.context_checkpointer", "ContextCheckpointerConfig"),
    "CoreTrainingLoop": ("softmax.training.rl.training.core", "CoreTrainingLoop"),
    "DistributedHelper": ("softmax.training.rl.training.distributed_helper", "DistributedHelper"),
    "EnvironmentMetaData": ("softmax.training.rl.training.training_environment", "EnvironmentMetaData"),
    "Evaluator": ("softmax.training.rl.training.evaluator", "Evaluator"),
    "EvaluatorConfig": ("softmax.training.rl.training.evaluator", "EvaluatorConfig"),
    "Experience": ("softmax.training.rl.training.experience", "Experience"),
    "GradientReporter": ("softmax.training.rl.training.gradient_reporter", "GradientReporter"),
    "GradientReporterConfig": ("softmax.training.rl.training.gradient_reporter", "GradientReporterConfig"),
    "Heartbeat": ("softmax.training.rl.training.heartbeat", "Heartbeat"),
    "HeartbeatConfig": ("softmax.training.rl.training.heartbeat", "HeartbeatConfig"),
    "HyperparameterSchedulerConfig": ("softmax.training.rl.training.scheduler", "HyperparameterSchedulerConfig"),
    "Monitor": ("softmax.training.rl.training.monitor", "Monitor"),
    "NoOpEvaluator": ("softmax.training.rl.training.evaluator", "NoOpEvaluator"),
    "NoOpStatsReporter": ("softmax.training.rl.training.stats_reporter", "NoOpStatsReporter"),
    "ProgressLogger": ("softmax.training.rl.training.progress_logger", "ProgressLogger"),
    "RolloutResult": ("softmax.training.rl.training.core", "RolloutResult"),
    "Scheduler": ("softmax.training.rl.training.scheduler", "Scheduler"),
    "SchedulerConfig": ("softmax.training.rl.training.scheduler", "SchedulerConfig"),
    "StatsReporter": ("softmax.training.rl.training.stats_reporter", "StatsReporter"),
    "StatsReporterConfig": ("softmax.training.rl.training.stats_reporter", "StatsReporterConfig"),
    "StatsReporterState": ("softmax.training.rl.training.stats_reporter", "StatsReporterState"),
    "TorchProfiler": ("softmax.training.rl.training.torch_profiler", "TorchProfiler"),
    "TrainerCallback": ("softmax.training.rl.training.component", "TrainerCallback"),
    "TrainerComponent": ("softmax.training.rl.training.component", "TrainerComponent"),
    "TrainerState": ("softmax.training.rl.training.component_context", "TrainerState"),
    "TrainingEnvironment": ("softmax.training.rl.training.training_environment", "TrainingEnvironment"),
    "TrainingEnvironmentConfig": ("softmax.training.rl.training.training_environment", "TrainingEnvironmentConfig"),
    "TrainingEnvWindow": ("softmax.training.rl.training.component_context", "TrainingEnvWindow"),
    "Uploader": ("softmax.training.rl.training.uploader", "Uploader"),
    "UploaderConfig": ("softmax.training.rl.training.uploader", "UploaderConfig"),
    "VectorizedTrainingEnvironment": (
        "softmax.training.rl.training.training_environment",
        "VectorizedTrainingEnvironment",
    ),
    "WandbAborter": ("softmax.training.rl.training.wandb_aborter", "WandbAborter"),
    "WandbAborterConfig": ("softmax.training.rl.training.wandb_aborter", "WandbAborterConfig"),
    "WandbLogger": ("softmax.training.rl.training.wandb_logger", "WandbLogger"),
}

# Explicitly define __all__ to help type checkers
__all__ = [
    "Checkpointer",
    "CheckpointerConfig",
    "ComponentContext",
    "ContextCheckpointer",
    "ContextCheckpointerConfig",
    "CoreTrainingLoop",
    "DistributedHelper",
    "EnvironmentMetaData",
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
]


def __getattr__(name: str) -> Any:
    """Dynamically import training submodules on attribute access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'softmax.training.rl.training' has no attribute '{name}'")

    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    attr = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
