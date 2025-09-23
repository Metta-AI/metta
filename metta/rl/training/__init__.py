"""Lazy-loading facade for training components.

This avoids import-time circular dependencies between loss modules and the
training package by deferring heavy submodule imports until the symbols are
actually needed. External callers can continue to use
``from metta.rl.training import Evaluator`` and similar statements without
change.
"""

import importlib
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str | None]] = {
    "Checkpointer": ("metta.rl.training.checkpointer", "Checkpointer"),
    "CheckpointerConfig": ("metta.rl.training.checkpointer", "CheckpointerConfig"),
    "ComponentContext": ("metta.rl.training.component_context", "ComponentContext"),
    "ContextCheckpointer": ("metta.rl.training.context_checkpointer", "ContextCheckpointer"),
    "ContextCheckpointerConfig": ("metta.rl.training.context_checkpointer", "ContextCheckpointerConfig"),
    "CoreTrainingLoop": ("metta.rl.training.core", "CoreTrainingLoop"),
    "DistributedHelper": ("metta.rl.training.distributed_helper", "DistributedHelper"),
    "EnvironmentMetaData": ("metta.rl.training.training_environment", "EnvironmentMetaData"),
    "Evaluator": ("metta.rl.training.evaluator", "Evaluator"),
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
}

__all__ = list(_EXPORTS.keys())  # type: ignore[reportUnsupportedDunderAll]


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
