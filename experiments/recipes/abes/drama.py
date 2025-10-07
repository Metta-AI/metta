from typing import Optional

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum as _make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
)
from metta.agent.policies.drama_policy import DramaPolicyConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import OptimizerConfig, TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool

make_curriculum = _make_curriculum

DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024
DEFAULT_PROFILE_INTERVAL_EPOCHS = 1
DEFAULT_PROFILE_DIR = "${run_dir}/torch_traces"


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
    profile_interval_epochs: int = DEFAULT_PROFILE_INTERVAL_EPOCHS,
    profile_dir: str = DEFAULT_PROFILE_DIR,
) -> TrainTool:
    """Train Drama policy in the arena basic easy shaped setup."""

    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    policy = policy_architecture or DramaPolicyConfig()

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        batch_size=batch_size,
        minibatch_size=minibatch_size,
    )

    training_env_cfg = TrainingEnvironmentConfig(
        curriculum=curriculum,
        forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    )

    evaluator_cfg = EvaluatorConfig(simulations=simulations())

    torch_profiler_cfg = TorchProfilerConfig(
        interval_epochs=profile_interval_epochs,
        profile_dir=profile_dir,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env_cfg,
        evaluator=evaluator_cfg,
        policy_architecture=policy,
        torch_profiler=torch_profiler_cfg,
    )


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
