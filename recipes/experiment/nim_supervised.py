"""Supervised trainer recipe that uses the Nim scripted agent as a teacher."""

from __future__ import annotations

from typing import Literal

from metta.cogworks.curriculum import env_curriculum
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import EnvSupervisorConfig, MettaGridConfig

NIM_TEACHER_POLICY = "nim_thinky"


def _nim_arena(num_agents: int, max_steps: int) -> MettaGridConfig:
    env = make_arena(num_agents=num_agents)
    env.game.max_steps = max_steps
    return env


def train(
    num_agents: int = 6,
    max_steps: int = 96,
    total_timesteps: int = 262_144,
    vectorization: Literal["serial", "multiprocessing"] = "serial",
) -> TrainTool:
    """Train via supervised imitation from the Nim scripted policy."""

    if num_agents % 6 != 0:
        msg = "MettaGrid curriculum currently expects agent counts divisible by 6"
        raise ValueError(f"{msg}; received num_agents={num_agents}")

    env_cfg = _nim_arena(num_agents=num_agents, max_steps=max_steps)
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
        evaluator=EvaluatorConfig(simulations=[SimulationConfig(suite="arena", name="nim_supervised", env=eval_env)]),
    )

    tool.trainer.behavior_cloning.policy_uri = NIM_TEACHER_POLICY
    tool.trainer.total_timesteps = total_timesteps
    tool.trainer.minibatch_size = 512
    tool.trainer.batch_size = 4096
    tool.trainer.bptt_horizon = 16

    tool.wandb.enabled = False
    tool.system.vectorization = vectorization
    tool.system.device = "cpu"

    # Fast local defaults.
    tool.evaluator.epoch_interval = 0
    tool.checkpointer.epoch_interval = 1
    tool.training_env.seed = 0

    return tool
