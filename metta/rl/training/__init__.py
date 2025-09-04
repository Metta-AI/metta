"""Training components for Metta RL."""

from metta.rl.training import vecenv
from metta.rl.training.component import ComponentConfig, MasterComponent, TrainingComponent
from metta.rl.training.core import CoreTrainingLoop, RolloutResult
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.evaluator import EvaluationConfig, Evaluator, NoOpEvaluator
from metta.rl.training.gradient_stats import GradientStatsComponent, GradientStatsConfig
from metta.rl.training.heartbeat import HeartbeatConfig, HeartbeatWriter
from metta.rl.training.hyperparameter import HyperparameterComponent, HyperparameterConfig
from metta.rl.training.policy_checkpointer import PolicyCheckpointer, PolicyCheckpointerConfig
from metta.rl.training.policy_uploader import PolicyUploader, PolicyUploaderConfig
from metta.rl.training.stats_reporter import NoOpStatsReporter, StatsConfig, StatsReporter, StatsState
from metta.rl.training.torch_profiler_component import TorchProfilerComponent, TorchProfilerConfig
from metta.rl.training.trainer_checkpointer import TrainerCheckpointer, TrainerCheckpointerConfig

__all__ = [
    # Core training
    "CoreTrainingLoop",
    "RolloutResult",
    # Distributed
    "DistributedHelper",
    # Checkpointing
    "PolicyCheckpointer",
    "PolicyCheckpointerConfig",
    "PolicyUploader",
    "PolicyUploaderConfig",
    "TrainerCheckpointer",
    "TrainerCheckpointerConfig",
    # Evaluation
    "Evaluator",
    "EvaluationConfig",
    "NoOpEvaluator",
    # Stats
    "StatsReporter",
    "StatsConfig",
    "StatsState",
    "NoOpStatsReporter",
    # Components
    "TrainingComponent",
    "MasterComponent",
    "ComponentConfig",
    # Torch profiler
    "TorchProfilerComponent",
    "TorchProfilerConfig",
    # Heartbeat
    "HeartbeatWriter",
    "HeartbeatConfig",
    # Hyperparameter
    "HyperparameterComponent",
    "HyperparameterConfig",
    # Gradient stats
    "GradientStatsComponent",
    "GradientStatsConfig",
    # Vecenv
    "vecenv",
]
