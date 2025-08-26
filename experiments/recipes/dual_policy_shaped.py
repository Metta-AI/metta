"""
Improved dual policy training with shaped rewards and continuous tracking.

This fixes the flat graph issue by:
1. Using shaped rewards for continuous feedback
2. Reducing max_steps to align with training frequency
3. Implementing continuous reward tracking

Usage:
    # Local test:
    python tools/run.py experiments.recipes.dual_policy_shaped:test

    # Skypilot training:
    devops/skypilot/launch.py experiments.recipes.dual_policy_shaped:train run=dual_policy_shaped
"""

import os
from typing import List, Optional

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import (
    CheckpointConfig,
    DualPolicyConfig,
    EvaluationConfig,
    TrainerConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.recipes.arena_basic_easy_shaped import make_env as make_shaped_env

# The NPC checkpoint you want to test with
NPC_CHECKPOINT = os.environ.get(
    "NPC_CHECKPOINT",
    "wandb://metta-research/metta/model/krishna_arena_1b:v22",
)


def _wandb_to_training_uri(uri: str) -> str:
    """Convert wandb selector URI to qualified artifact."""
    try:
        if not uri.startswith("wandb://"):
            return uri
        rest = uri[len("wandb://") :]
        version = None
        if ":" in rest:
            rest, version = rest.split(":", 1)
        parts = rest.split("/")
        if len(parts) == 4:
            entity, project, _artifact_type, name = parts
            base = f"wandb://{entity}/{project}/{name}"
            return f"{base}:{version}" if version else base
        return uri
    except Exception:
        return uri


def make_env(num_agents: int = 24) -> EnvConfig:
    """Create shaped reward environment with proper episode length."""
    env = make_shaped_env(num_agents=num_agents)

    # CRITICAL FIX: Reduce max_steps to align with training frequency
    # With batch_size=524288, bptt_horizon=64, and 24 agents:
    # Training happens every ~341 steps, so episodes should be shorter
    env.game.max_steps = 256  # Episodes complete more frequently

    return env


def make_curriculum(env: Optional[EnvConfig] = None) -> CurriculumConfig:
    """Create a simple curriculum for dual policy training."""
    env = env or make_env()
    return cc.env_curriculum(env)


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations for dual policy setup."""
    basic_env = make_env(num_agents=24)

    # Create evaluations with shaped rewards
    return [
        SimulationConfig(
            name="dual_policy/shaped",
            env=basic_env,
            npc_policy_uri=NPC_CHECKPOINT,
            policy_agents_pct=0.5,  # 50% training, 50% NPC
        ),
        # You can add more evaluation scenarios here
    ]


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """
    Main training function with fixes for continuous reward tracking.

    Key improvements:
    1. Shaped rewards provide continuous feedback
    2. Shorter episodes align with training frequency
    3. More frequent evaluations for better monitoring
    """
    env = make_env()

    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(env),
        total_timesteps=10_000_000_000,  # 10B timesteps
        # Dual policy configuration
        dual_policy=DualPolicyConfig(
            enabled=True,
            training_agents_pct=0.5,  # 50% training, 50% NPC
            checkpoint_npc=_wandb_to_training_uri(NPC_CHECKPOINT),
        ),
        # Checkpoint configuration
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,
            wandb_checkpoint_interval=50,
        ),
        # More frequent evaluation for better monitoring
        evaluation=EvaluationConfig(
            simulations=make_evals(env),
            evaluate_remote=True,
            evaluate_local=False,
            evaluate_interval=500,  # More frequent evaluation
        ),
        # Batch configuration
        # Keeping defaults but you could adjust these
        # batch_size=524288,  # Default
        # minibatch_size=16384,  # Default
        # bptt_horizon=64,  # Default
    )

    return TrainTool(trainer=trainer_cfg)


def test(total_timesteps: int = 100_000) -> TrainTool:
    """
    Quick test function for local development.
    """
    env = make_env()

    trainer_cfg = TrainerConfig(
        curriculum=make_curriculum(env),
        total_timesteps=total_timesteps,
        # Dual policy configuration
        dual_policy=DualPolicyConfig(
            enabled=True,
            training_agents_pct=0.5,
            checkpoint_npc=_wandb_to_training_uri(NPC_CHECKPOINT),
        ),
        # Enable evaluations for testing
        evaluation=EvaluationConfig(
            simulations=make_evals(env),
            skip_git_check=True,
            evaluate_interval=50,  # Very frequent for testing
            evaluate_local=True,
            evaluate_remote=False,
        ),
        # Minimal settings for fast testing
        batch_size=2048,
        minibatch_size=256,
        rollout_workers=4,
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    """Interactive play tool."""
    eval_env = env or make_env()
    return PlayTool(sim=SimulationConfig(env=eval_env, name="dual_policy_shaped"))


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay tool."""
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="dual_policy_shaped"))


def evaluate(
    policy_uri: str, simulations: Optional[List[SimulationConfig]] = None
) -> SimTool:
    """Evaluate a trained policy."""
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
