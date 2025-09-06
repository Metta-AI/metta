from typing import Optional

from metta.rl.trainer_config import TrainerConfig, EvaluationConfig, CheckpointConfig
from metta.rl.loss.loss_config import LossConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sim.env_config import TribalEnvConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_tribal_environment(
    max_steps: int = 2000,
    enable_combat: bool = True,
    **kwargs
) -> TribalEnvConfig:
    """
    Create tribal environment configuration for training.
    
    The tribal environment features:
    - Village-based agent tribes with shared altars (15 agents, compile-time constant)
    - Multi-step resource chains (ore → battery → hearts)
    - Crafting system (wood → spears, wheat → hats/food, ore → armor)
    - Defensive gameplay against Clippy enemies
    - Terrain interaction (water, wheat fields, forests)
    
    NOTE: Agent count, map dimensions, and observation space are compile-time constants
    for performance. Only gameplay parameters are configurable.
    """
    config = TribalEnvConfig(
        label="tribal_basic",
        desync_episodes=True,
        **kwargs
    )
    
    # Configure game mechanics (only runtime-configurable parameters)
    config.game.max_steps = max_steps
    config.game.enable_combat = enable_combat
    
    # Set up resource chain rewards
    config.game.heart_reward = 10.0  # High reward for completing the chain
    config.game.battery_reward = 0.01  # Medium reward for intermediate product (match original)
    config.game.ore_reward = 0.003     # Small reward for raw materials (match original)
    
    return config


def train() -> TrainTool:
    """
    Train agents on the tribal environment.
    
    Uses a minimal configuration similar to the working arena recipe.
    """
    # Create environment (uses compile-time constant: 15 agents)
    env = make_tribal_environment()
    
    # Minimal trainer config like arena recipe
    trainer_config = TrainerConfig(
        losses=LossConfig(),
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(name="tribal/basic", env=env),
            ],
            skip_git_check=True,  # Skip git check for development
        ),
    )
    
    return TrainTool(trainer=trainer_config)


def evaluate(
    policy_uri: str,
    run: str = "tribal_eval",
    num_episodes: int = 10,
    **overrides
) -> SimTool:
    """
    Evaluate a trained policy on the tribal environment.
    
    Args:
        policy_uri: URI to trained policy (file:// or wandb://)
        run: Name for this evaluation run  
        num_episodes: Number of episodes to evaluate
        **overrides: Additional configuration overrides
    """
    env = make_tribal_environment()
    
    return SimTool(
        env=env,
        policy_uri=policy_uri,
        num_episodes=num_episodes,
        run=run,
        **overrides
    )


def play(
    policy_uri: str,
    render_mode: str = "human",
    **overrides
) -> PlayTool:
    """
    Interactive play with a trained policy.
    
    Args:
        policy_uri: URI to trained policy
        render_mode: Rendering mode for visualization
        **overrides: Additional configuration overrides
    """
    env = make_tribal_environment(render_mode=render_mode)
    
    return PlayTool(
        env=env,
        policy_uri=policy_uri,
        **overrides
    )


def replay(
    policy_uri: str,
    **overrides
) -> ReplayTool:
    """
    Replay recorded tribal episodes.
    
    Args:
        policy_uri: URI to policy that generated replays
        **overrides: Additional configuration overrides
    """
    return ReplayTool(
        policy_uri=policy_uri,
        **overrides
    )
