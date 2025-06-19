"""Environment factory functions for creating environments without Hydra."""

from typing import Any, Dict, List, Optional

from pufferlib.vectorization import PettingZooVectorization

from mettagrid import MettaGridEnv
from mettagrid.curriculum import Curriculum, CurriculumWrapper


def create_env(
    width: int = 11,
    height: int = 11,
    max_steps: int = 1024,
    num_agents: int = 1,
    num_resources: int = 1,
    resource_prob: float = 0.01,
    **kwargs,
) -> MettaGridEnv:
    """Create a MettaGrid environment with specified parameters.

    Args:
        width: Width of the grid
        height: Height of the grid
        max_steps: Maximum steps per episode
        num_agents: Number of agents in the environment
        num_resources: Number of resource types
        resource_prob: Probability of resource spawning
        **kwargs: Additional environment parameters

    Returns:
        Configured MettaGridEnv instance
    """
    env_config = {
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "num_agents": num_agents,
        "num_resources": num_resources,
        "resource_spawn_prob": resource_prob,
        **kwargs,
    }

    return MettaGridEnv(**env_config)


def create_curriculum_env(
    base_config: Optional[Dict[str, Any]] = None, curriculum_stages: Optional[List[Dict[str, Any]]] = None, **kwargs
) -> CurriculumWrapper:
    """Create an environment with curriculum learning.

    Args:
        base_config: Base environment configuration
        curriculum_stages: List of curriculum stage configurations
        **kwargs: Additional parameters

    Returns:
        Environment wrapped with curriculum
    """
    if base_config is None:
        base_config = {
            "width": 11,
            "height": 11,
            "max_steps": 1024,
        }

    if curriculum_stages is None:
        # Default simple curriculum
        curriculum_stages = [
            {
                "name": "easy",
                "duration": 100000,
                "env_config": {
                    "num_agents": 1,
                    "num_resources": 1,
                    "resource_spawn_prob": 0.02,
                },
            },
            {
                "name": "medium",
                "duration": 200000,
                "env_config": {
                    "num_agents": 2,
                    "num_resources": 2,
                    "resource_spawn_prob": 0.015,
                },
            },
            {
                "name": "hard",
                "duration": float("inf"),
                "env_config": {
                    "num_agents": 4,
                    "num_resources": 3,
                    "resource_spawn_prob": 0.01,
                },
            },
        ]

    # Create base environment
    base_env = create_env(**base_config, **kwargs)

    # Create curriculum
    curriculum = Curriculum(stages=curriculum_stages)

    # Wrap with curriculum
    return CurriculumWrapper(base_env, curriculum)


def create_vectorized_env(
    env_fn=None, num_envs: int = 64, num_workers: int = 4, device: str = "cuda", **env_kwargs
) -> PettingZooVectorization:
    """Create a vectorized environment for parallel training.

    Args:
        env_fn: Environment factory function (defaults to create_env)
        num_envs: Total number of parallel environments
        num_workers: Number of worker processes
        device: Device for tensor operations
        **env_kwargs: Arguments passed to env_fn

    Returns:
        Vectorized environment
    """
    if env_fn is None:
        env_fn = lambda: create_env(**env_kwargs)

    return PettingZooVectorization(
        env_creator=env_fn,
        num_envs=num_envs,
        num_workers=num_workers,
        device=device,
    )


# Pre-configured environment presets
ENV_PRESETS = {
    "tiny": {
        "width": 5,
        "height": 5,
        "max_steps": 256,
        "num_agents": 1,
    },
    "small": {
        "width": 7,
        "height": 7,
        "max_steps": 512,
        "num_agents": 1,
    },
    "medium": {
        "width": 11,
        "height": 11,
        "max_steps": 1024,
        "num_agents": 2,
    },
    "large": {
        "width": 15,
        "height": 15,
        "max_steps": 2048,
        "num_agents": 4,
    },
    "huge": {
        "width": 21,
        "height": 21,
        "max_steps": 4096,
        "num_agents": 8,
    },
}


def create_env_from_preset(preset_name: str, **override_kwargs) -> MettaGridEnv:
    """Create an environment from a predefined preset.

    Args:
        preset_name: Name of the preset (tiny, small, medium, large, huge)
        **override_kwargs: Parameters to override from the preset

    Returns:
        Configured environment
    """
    if preset_name not in ENV_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(ENV_PRESETS.keys())}")

    config = ENV_PRESETS[preset_name].copy()
    config.update(override_kwargs)

    return create_env(**config)
