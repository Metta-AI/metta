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
    # Core training
    "CoreTrainingLoop": ("metta.rl.training.core", "CoreTrainingLoop"),
    "RolloutResult": ("metta.rl.training.core", "RolloutResult"),
    # Distributed helpers
    "DistributedHelper": ("metta.rl.training.distributed_helper", "DistributedHelper"),
    # Checkpointing components
    "PolicyCheckpointer": ("metta.rl.training.policy_checkpointer", "PolicyCheckpointer"),
    "PolicyCheckpointerConfig": ("metta.rl.training.policy_checkpointer", "PolicyCheckpointerConfig"),
    "PolicyUploader": ("metta.rl.training.policy_uploader", "PolicyUploader"),
    "PolicyUploaderConfig": ("metta.rl.training.policy_uploader", "PolicyUploaderConfig"),
    "TrainerCheckpointer": ("metta.rl.training.trainer_checkpointer", "TrainerCheckpointer"),
    "TrainerCheckpointerConfig": ("metta.rl.training.trainer_checkpointer", "TrainerCheckpointerConfig"),
    "TrainerContext": ("metta.rl.training.context", "TrainerContext"),
    "WandbAbortComponent": ("metta.rl.training.wandb_abort", "WandbAbortComponent"),
    # Evaluation
    "Evaluator": ("metta.rl.training.evaluator", "Evaluator"),
    "EvaluatorConfig": ("metta.rl.training.evaluator", "EvaluatorConfig"),
    "NoOpEvaluator": ("metta.rl.training.evaluator", "NoOpEvaluator"),
    # Stats reporting
    "StatsReporter": ("metta.rl.training.stats_reporter", "StatsReporter"),
    "StatsConfig": ("metta.rl.training.stats_reporter", "StatsConfig"),
    "StatsState": ("metta.rl.training.stats_reporter", "StatsState"),
    "NoOpStatsReporter": ("metta.rl.training.stats_reporter", "NoOpStatsReporter"),
    # Trainer components
    "TrainerComponent": ("metta.rl.training.component", "TrainerComponent"),
    "MonitoringComponent": ("metta.rl.training.monitoring_component", "MonitoringComponent"),
    # Torch profiler
    "TorchProfilerComponent": ("metta.rl.training.torch_profiler_component", "TorchProfilerComponent"),
    # Heartbeat
    "HeartbeatWriter": ("metta.rl.training.heartbeat", "HeartbeatWriter"),
    "HeartbeatConfig": ("metta.rl.training.heartbeat", "HeartbeatConfig"),
    # Hyperparameter scheduler
    "HyperparameterComponent": ("metta.rl.training.hyperparameter", "HyperparameterComponent"),
    "HyperparameterConfig": ("metta.rl.training.hyperparameter", "HyperparameterConfig"),
    "HyperparameterSchedulerConfig": ("metta.rl.training.hyperparameter", "HyperparameterSchedulerConfig"),
    # Gradient stats
    "GradientStatsComponent": ("metta.rl.training.gradient_stats", "GradientStatsComponent"),
    "GradientStatsConfig": ("metta.rl.training.gradient_stats", "GradientStatsConfig"),
    # Training environment module
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
