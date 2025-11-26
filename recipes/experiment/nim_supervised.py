"""Supervised trainer recipe aligned with the extractor_hub_30 lonely_heart mission."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

from cogames.cli.mission import get_all_eval_missions, get_all_missions, get_mission
from metta.cogworks.curriculum import CurriculumConfig, env_curriculum, merge, single_task
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import EnvSupervisorConfig, MettaGridConfig


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
    mission: str = "evals.extractor_hub_30",
    variants: Iterable[str] | None = ("lonely_heart",),
    cogs: int = 1,
    max_steps: int = 1000,
    total_timesteps: int = 400_000,
    vectorization: Literal["serial", "multiprocessing"] = "serial",
    resume_policy_uri: str | None = None,
    learning_rate: float = 0.001153637 * 1,
) -> TrainTool:
    """Train via supervised imitation from the Nim scripted policy."""

    # decide if we want to just train on a single simple mission for debugging instead of using
    # a curriculum over all cogames missions
    simple_mission = True
    if simple_mission:
        env_config = _load_env_from_mission(mission, tuple(variants) if variants else None, cogs, max_steps)
        curriculum = env_curriculum(env_config)
        eval_env = env_config
    else:
        curriculum = create_cogames_curriculum(variants=variants, cogs=cogs, max_steps=max_steps)
        eval_env = _load_env_from_mission(mission, tuple(variants) if variants else None, cogs, max_steps)

    tool = TrainTool(
        training_env=TrainingEnvironmentConfig(
            curriculum=curriculum,
            supervisor=EnvSupervisorConfig(policy="nim_thinky"),
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=1024,
            vectorization=vectorization,
        ),
        evaluator=EvaluatorConfig(simulations=[SimulationConfig(suite=mission, name="nim_supervised", env=eval_env)]),
    )

    # tool.trainer.behavior_cloning.policy_uri = "nim_thinky"
    # tool.trainer.behavior_cloning.student_led = False

    tool.trainer.total_timesteps = total_timesteps
    tool.trainer.minibatch_size = 16_384
    tool.trainer.batch_size = 524_288
    tool.trainer.bptt_horizon = 64
    tool.trainer.optimizer.learning_rate = learning_rate

    tool.trainer.losses.supervisor.teacher_random_walk_prob = 0.0
    tool.trainer.losses.supervisor.teacher_lead_prob = 0.9
    tool.trainer.losses.supervisor.enabled = True
    tool.trainer.losses.ppo_actor.enabled = False

    tool.wandb.enabled = False
    tool.system.vectorization = vectorization
    tool.system.device = "cpu"

    # generate replays during training
    # tool.training_env.write_replays = True

    # Fast local defaults.
    tool.evaluator.epoch_interval = 0
    tool.checkpointer.epoch_interval = 1
    tool.training_env.seed = 0

    tool.evaluator.epoch_interval = 0

    # tool.initial_policy_uri = "train_dir/local.dffarr.20251121.151430/checkpoints/local.df...

    return tool


def create_cogames_curriculum(
    *,
    variants: Iterable[str] | None = None,
    cogs: int = 1,
    max_steps: int = 1000,
) -> CurriculumConfig:
    """Create a curriculum that contains all the cogames missions (except for evals).

    Args:
        variants: Optional variants to apply to all missions
        cogs: Number of cogs (agents) for each mission
        max_steps: Maximum steps per episode for each mission

    Returns:
        A CurriculumConfig that rotates through all non-eval missions
    """
    all_missions = get_all_missions()
    eval_missions = set(get_all_eval_missions())

    training_missions = [m for m in all_missions if m not in eval_missions]

    variant_tuple = tuple(variants) if variants else None
    task_generators = []

    for mission_name in training_missions:
        env_cfg = _load_env_from_mission(mission_name, variant_tuple, cogs, max_steps)
        task_generators.append(single_task(env_cfg))

    if len(task_generators) == 1:
        return task_generators[0].to_curriculum()
    else:
        merged = merge(task_generators)
        return merged.to_curriculum()
