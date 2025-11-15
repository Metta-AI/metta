"""Supervised trainer recipe aligned with the extractor_hub_30 lonely_heart mission."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

from cogames.cli.mission import get_mission
from metta.cogworks.curriculum import env_curriculum
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import EnvSupervisorConfig, MettaGridConfig

NIM_TEACHER_POLICY = "nim_thinky"
DEFAULT_MISSION = "evals.extractor_hub_30"
DEFAULT_VARIANTS: tuple[str, ...] = ("lonely_heart",)
SUPERVISED_RESUME_POLICY_URI: str | None = "file://local.root.20251115.022755:v95.mpt"


def _load_env_from_mission(
    mission: str,
    variants: Sequence[str] | None,
    cogs: int,
    max_steps: int,
) -> MettaGridConfig:
    variant_list = list(variants) if variants else None
    _, env_cfg, _ = get_mission(mission, variants_arg=variant_list, cogs=cogs)
    env_cfg.game.max_steps = max_steps
    return env_cfg


def train(
    *,
    mission: str = DEFAULT_MISSION,
    variants: Iterable[str] | None = DEFAULT_VARIANTS,
    cogs: int = 1,
    max_steps: int = 1000,
    total_timesteps: int = 262_144,
    vectorization: Literal["serial", "multiprocessing"] = "serial",
    resume_policy_uri: str | None = SUPERVISED_RESUME_POLICY_URI,
) -> TrainTool:
    """Train via supervised imitation from the Nim scripted policy."""

    env_cfg = _load_env_from_mission(mission, tuple(variants) if variants else None, cogs, max_steps)
    curriculum = env_curriculum(env_cfg)
    eval_env = env_cfg.model_copy(deep=True)

    tool = TrainTool(
        training_env=TrainingEnvironmentConfig(
            curriculum=curriculum,
            supervisor=EnvSupervisorConfig(policy=NIM_TEACHER_POLICY),
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=1024,
            vectorization=vectorization,
        ),
        evaluator=EvaluatorConfig(simulations=[SimulationConfig(suite=mission, name="nim_supervised", env=eval_env)]),
    )

    tool.trainer.behavior_cloning.policy_uri = NIM_TEACHER_POLICY
    tool.trainer.behavior_cloning.student_led = True

    tool.trainer.total_timesteps = total_timesteps
    tool.trainer.minibatch_size = 512
    tool.trainer.batch_size = 4096
    tool.trainer.bptt_horizon = 16

    tool.trainer.losses.supervisor.teacher_random_walk_prob = 0.2

    tool.wandb.enabled = False
    tool.system.vectorization = vectorization
    tool.system.device = "cpu"

    # Fast local defaults.
    tool.evaluator.epoch_interval = 0
    tool.checkpointer.epoch_interval = 1
    tool.training_env.seed = 0

    if resume_policy_uri:
        tool.initial_policy_uri = resume_policy_uri

    return tool
