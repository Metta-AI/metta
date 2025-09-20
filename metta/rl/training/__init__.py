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
    "CoreTrainingLoop": ("metta.rl.training.core", "CoreTrainingLoop"),
    "RolloutResult": ("metta.rl.training.core", "RolloutResult"),
    "DistributedHelper": ("metta.rl.training.distributed_helper", "DistributedHelper"),
    "Checkpointer": ("metta.rl.training.checkpointer", "Checkpointer"),
    "CheckpointerConfig": ("metta.rl.training.checkpointer", "CheckpointerConfig"),
    "Uploader": ("metta.rl.training.uploader", "Uploader"),
    "UploaderConfig": ("metta.rl.training.uploader", "UploaderConfig"),
    "ContextCheckpointer": ("metta.rl.training.context_checkpointer", "ContextCheckpointer"),
    "ContextCheckpointerConfig": ("metta.rl.training.context_checkpointer", "ContextCheckpointerConfig"),
    "ComponentContext": ("metta.rl.training.component_context", "ComponentContext"),
    "TrainerState": ("metta.rl.training.component_context", "TrainerState"),
    "TrainingEnvWindow": ("metta.rl.training.component_context", "TrainingEnvWindow"),
    "WandbAborter": ("metta.rl.training.wandb_aborter", "WandbAborter"),
    "WandbAborterConfig": ("metta.rl.training.wandb_aborter", "WandbAborterConfig"),
    "Evaluator": ("metta.rl.training.evaluator", "Evaluator"),
    "EvaluatorConfig": ("metta.rl.training.evaluator", "EvaluatorConfig"),
    "NoOpEvaluator": ("metta.rl.training.evaluator", "NoOpEvaluator"),
    "StatsReporter": ("metta.rl.training.stats_reporter", "StatsReporter"),
    "StatsReporterConfig": ("metta.rl.training.stats_reporter", "StatsReporterConfig"),
    "StatsReporterState": ("metta.rl.training.stats_reporter", "StatsReporterState"),
    "NoOpStatsReporter": ("metta.rl.training.stats_reporter", "NoOpStatsReporter"),
    "TrainerComponent": ("metta.rl.training.component", "TrainerComponent"),
    "Monitor": ("metta.rl.training.monitor", "Monitor"),
    "TorchProfiler": ("metta.rl.training.torch_profiler", "TorchProfiler"),
    "Heartbeat": ("metta.rl.training.heartbeat", "Heartbeat"),
    "HeartbeatConfig": ("metta.rl.training.heartbeat", "HeartbeatConfig"),
    "Scheduler": ("metta.rl.training.scheduler", "Scheduler"),
    "SchedulerConfig": ("metta.rl.training.scheduler", "SchedulerConfig"),
    "HyperparameterSchedulerConfig": ("metta.rl.training.scheduler", "HyperparameterSchedulerConfig"),
    "GradientReporter": ("metta.rl.training.gradient_reporter", "GradientReporter"),
    "GradientReporterConfig": ("metta.rl.training.gradient_reporter", "GradientReporterConfig"),
    "ProgressLogger": ("metta.rl.training.progress_logger", "ProgressLogger"),
    "training_environment": ("metta.rl.training.training_environment", None),
}

__all__ = list(_EXPORTS.keys())


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
