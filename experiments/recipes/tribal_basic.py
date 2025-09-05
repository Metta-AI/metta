"""
Tribal Environment Training Recipe

This recipe configures training for the Nim tribal environment
with village-based cooperation, resource management, and defense mechanics.
"""

from typing import Optional

from metta.rl.trainer_config import TrainerConfig
from metta.sim.tribal_genny import make_tribal_env, TribalGridEnv
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_tribal_environment(
    num_agents: int = 15,
    max_steps: int = 2000,
    **kwargs
) -> TribalGridEnv:
    """
    Create tribal environment with optimized settings for RL training.
    
    The tribal environment features:
    - Village-based agent tribes with shared altars
    - Multi-step resource chains (ore → battery → hearts)
    - Crafting system (wood → spears, wheat → hats/food, ore → armor)
    - Defensive gameplay against Clippy enemies
    - Terrain interaction (water, wheat fields, forests)
    """
    return make_tribal_env(
        num_agents=num_agents,
        max_steps=max_steps,
        render_mode=None,  # No rendering during training
        **kwargs
    )


def train(
    run: str = "tribal_basic",
    num_agents: int = 15,
    total_timesteps: int = 1_000_000,
    num_workers: int = 4,
    **overrides
) -> TrainTool:
    """
    Train agents on the tribal environment.
    
    Args:
        run: Name for this training run
        num_agents: Number of agents per environment
        total_timesteps: Total training timesteps
        num_workers: Number of parallel workers
        **overrides: Additional configuration overrides
    """
    # Create environment
    env = make_tribal_environment(num_agents=num_agents)
    
    # Configure trainer for tribal environment
    trainer_config = TrainerConfig(
        total_timesteps=total_timesteps,
        rollout_workers=num_workers,
        
        # Tribal-specific tuning
        batch_size=2048,  # Larger batch for stable multi-agent learning
        minibatch_size=256,
        learning_rate=3e-4,
        
        # Episode settings
        max_episode_length=2000,  # Longer episodes for complex strategies
        
        # Evaluation
        evaluation=TrainerConfig.EvaluationConfig(
            enabled=True,
            interval=50000,  # Evaluate every 50k steps
            episodes=10,
        ),
        
        # Checkpointing
        checkpoint=TrainerConfig.CheckpointConfig(
            enabled=True,
            interval=100000,  # Save every 100k steps
            save_on_exit=True,
        ),
        
        # Skip git check for development
        simulation=TrainerConfig.SimulationConfig(
            skip_git_check=True,
        ),
    )
    
    return TrainTool(
        env=env,
        trainer_config=trainer_config,
        run=run,
        **overrides
    )


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