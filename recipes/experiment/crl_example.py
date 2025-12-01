"""Example recipe for Contrastive RL (CRL) training.

Contrastive RL is a self-supervised goal-conditioned RL algorithm based on:
"1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable
New Goal-Reaching Capabilities" (Wang et al., 2025)

Key features:
- Uses InfoNCE contrastive loss to learn temporal distance functions
- Trains very deep residual networks (up to 1024 layers)
- No explicit reward function - the critic output IS the reward signal
- Goal-conditioned: agent learns to reach commanded goal states

Usage:
    # Train with CRL (64 layers, default settings)
    uv run ./tools/run.py train experiment/crl_example run=crl_test trainer.total_timesteps=100000

    # Train with deeper networks (256 layers)
    uv run ./tools/run.py train experiment/crl_example:train_deep run=crl_deep

    # Train CRL alongside PPO (hybrid approach)
    uv run ./tools/run.py train experiment/crl_example:train_hybrid run=crl_hybrid
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.rl.loss.crl import CRLConfig
from metta.rl.loss.crl_actor import CRLActorConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.ppo_actor import PPOActorConfig
from metta.rl.loss.ppo_critic import PPOCriticConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def mettagrid(num_agents: int = 8) -> MettaGridConfig:
    """Create a simple arena environment for CRL training."""
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation simulations."""
    env = env or mettagrid()
    return [
        SimulationConfig(suite="crl", name="arena", env=env),
    ]


def crl_losses(
    depth: int = 64,
    hidden_dim: int = 256,
    embedding_dim: int = 64,
) -> LossesConfig:
    """Configure CRL losses with specified depth.

    Args:
        depth: Network depth (number of Dense layers). Must be multiple of 4.
            Paper recommends: 4, 8, 16, 32, 64, 256, or 1024
        hidden_dim: Width of residual blocks
        embedding_dim: Output embedding dimension

    Returns:
        LossesConfig with CRL enabled
    """
    return LossesConfig(
        # Disable PPO - we're using pure CRL
        ppo_actor=PPOActorConfig(enabled=False),
        ppo_critic=PPOCriticConfig(enabled=False),
        # Enable CRL
        crl=CRLConfig(
            enabled=True,
            depth=depth,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            temperature=0.1,  # InfoNCE temperature
            critic_lr=3e-4,
            critic_coef=1.0,
            discount=0.99,  # For geometric goal sampling
            logsumexp_penalty=0.1,
        ),
        crl_actor=CRLActorConfig(
            enabled=True,
            actor_coef=1.0,
            entropy_coef=0.01,
            discount=0.99,
            her_ratio=0.5,  # Hindsight experience replay ratio
        ),
    )


def hybrid_losses(depth: int = 32) -> LossesConfig:
    """Configure hybrid PPO + CRL losses.

    This combines PPO for standard RL with CRL for representation learning.
    Useful when you want the benefits of both approaches.
    """
    return LossesConfig(
        # Keep PPO enabled
        ppo_actor=PPOActorConfig(enabled=True),
        ppo_critic=PPOCriticConfig(enabled=True),
        # Also enable CRL as auxiliary loss
        crl=CRLConfig(
            enabled=True,
            depth=depth,
            critic_coef=0.5,  # Lower weight since PPO is primary
        ),
        crl_actor=CRLActorConfig(
            enabled=False,  # Use PPO actor instead
        ),
    )


def train(
    depth: int = 64,
    curriculum: Optional[cc.CurriculumConfig] = None,
) -> TrainTool:
    """Train with CRL (pure contrastive RL).

    Args:
        depth: Network depth for CRL critic (must be multiple of 4)
        curriculum: Optional curriculum config

    Returns:
        TrainTool configured for CRL
    """
    env = mettagrid()
    curriculum = curriculum or cc.env_curriculum(env)

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=TrainerConfig(
            losses=crl_losses(depth=depth),
            total_timesteps=50_000_000,
        ),
        evaluator=EvaluatorConfig(
            simulations=simulations(env),
            evaluate_local=True,
        ),
    )


def train_deep() -> TrainTool:
    """Train with very deep CRL networks (256 layers).

    This follows the paper's finding that deeper networks unlock
    qualitatively different behaviors and emergent capabilities.
    """
    return train(depth=256)


def train_shallow() -> TrainTool:
    """Train with shallow CRL networks (4 layers) as baseline."""
    return train(depth=4)


def train_hybrid() -> TrainTool:
    """Train with hybrid PPO + CRL approach.

    Uses PPO for the main RL objective while CRL provides
    auxiliary representation learning.
    """
    env = mettagrid()
    curriculum = cc.env_curriculum(env)

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=TrainerConfig(
            losses=hybrid_losses(depth=32),
            total_timesteps=50_000_000,
        ),
        evaluator=EvaluatorConfig(
            simulations=simulations(env),
            evaluate_local=True,
        ),
    )


def train_scaling_experiment() -> list[TrainTool]:
    """Generate configs for depth scaling experiment.

    Creates training configs for depths 4, 8, 16, 32, 64 to reproduce
    the paper's scaling experiments.
    """
    depths = [4, 8, 16, 32, 64]
    return [train(depth=d) for d in depths]
