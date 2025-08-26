"""
Dual policy training recipe for skypilot training with NPC agents.

Usage:
    # Local test:
    python tools/run.py experiments.recipes.dual_policy:test

    # Skypilot training:
    devops/skypilot/launch.py experiments.recipes.dual_policy:train run=dual_policy_test
"""

import os
from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
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

# The NPC checkpoint you want to test with
# Note: Wandb artifact names use entity/project/artifact:version format
# You can override this by setting an environment variable or passing it as an argument
NPC_CHECKPOINT = os.environ.get(
    "NPC_CHECKPOINT",
    "wandb://metta-research/metta/model/krishna_arena_1b:v24",
)


def _wandb_to_training_uri(uri: str) -> str:
    """Convert wandb selector URI (entity/project/type/name[:ver]) to qualified artifact (entity/project/name[:ver]).

    Training loader expects 3-part path; evaluation accepts 4-part.
    """
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
        # already qualified or other forms
        return uri
    except Exception:
        return uri


def make_env(num_agents: int = 24) -> EnvConfig:
    """Create the arena environment configuration."""
    return eb.make_arena(num_agents=num_agents)


def make_curriculum(env: Optional[EnvConfig] = None) -> CurriculumConfig:
    """Create a simple curriculum for dual policy training."""
    env = env or make_env()

    # For now, use a simple env curriculum
    # You can expand this with bucketed tasks like arena.py if needed
    return cc.env_curriculum(env)


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations for dual policy setup.

    Creates evaluations that will track metrics separately for:
    - NPC agents
    - Training agents
    - Combined performance
    """
    basic_env = eb.make_arena(num_agents=24, combat=False)  # laser cost = 100
    combat_env = eb.make_arena(num_agents=24, combat=True)  # laser cost = 1

    # Main dual policy evaluations with 50/50 split
    # The metrics system will automatically track npc, trained, and combined stats
    return [
        SimulationConfig(
            name="dual_policy/basic",
            env=basic_env,
            npc_policy_uri=NPC_CHECKPOINT,
            policy_agents_pct=0.5,  # 50% training, 50% NPC
        ),
        SimulationConfig(
            name="dual_policy/combat",
            env=combat_env,
            npc_policy_uri=NPC_CHECKPOINT,
            policy_agents_pct=0.5,  # 50% training, 50% NPC
        ),
    ]


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """
    Main training function for skypilot with dual policy configuration.

    This function returns a TrainTool configured for dual policy training
    where half the agents are controlled by an NPC checkpoint and half
    are training agents.

    Uses default parameters for batch configuration:
    - batch_size: 524288 (default)
    - minibatch_size: 16384 (default)
    - rollout_workers: 1 (default)
    """
    env = make_env()

    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(env),
        total_timesteps=10_000_000_000,  # 10B timesteps for full training
        # Dual policy configuration
        dual_policy=DualPolicyConfig(
            enabled=True,
            training_agents_pct=0.5,  # 50% training, 50% NPC
            checkpoint_npc=_wandb_to_training_uri(NPC_CHECKPOINT),
        ),
        # Checkpoint configuration for skypilot
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,
            wandb_checkpoint_interval=50,
        ),
        # Evaluation configuration
        evaluation=EvaluationConfig(
            simulations=make_evals(env),
            evaluate_remote=True,  # For skypilot
            evaluate_local=False,
            evaluate_interval=1000,  # Evaluate every 1000 iterations
        ),
        # All other parameters use defaults from TrainerConfig
    )

    return TrainTool(trainer=trainer_cfg)


def test(total_timesteps: int = 100_000) -> TrainTool:
    """
    Quick test function for local development.

    This is a lightweight version for testing dual policy locally
    before launching on skypilot.
    """
    env = make_env()

    trainer_cfg = TrainerConfig(
        curriculum=make_curriculum(env),
        total_timesteps=total_timesteps,  # Short test run
        # Dual policy configuration
        dual_policy=DualPolicyConfig(
            enabled=True,
            training_agents_pct=0.5,  # 50% training, 50% NPC
            checkpoint_npc=_wandb_to_training_uri(NPC_CHECKPOINT),
        ),
        # Enable evaluations for local testing to see dual policy metrics
        evaluation=EvaluationConfig(
            simulations=make_evals(env),
            skip_git_check=True,
            evaluate_interval=100,  # Evaluate every 100 iterations for quick feedback
            evaluate_local=True,  # Run evaluations locally
            evaluate_remote=False,  # Don't use remote evaluation for local testing
        ),
        # Minimal settings for fast testing
        batch_size=2048,
        minibatch_size=256,
        rollout_workers=4,
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    """Interactive play tool for testing the environment."""
    eval_env = env or make_env()
    return PlayTool(sim=SimulationConfig(env=eval_env, name="dual_policy"))


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay tool for viewing recorded games."""
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="dual_policy"))


def evaluate(
    policy_uri: str, simulations: Optional[List[SimulationConfig]] = None
) -> SimTool:
    """Evaluate a trained policy."""
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
