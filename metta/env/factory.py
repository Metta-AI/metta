"""Environment factory functions for creating Metta environments."""

from typing import Any

# Optional imports - not all may be available
try:
    from pufferlib.vectorization import PettingZooVectorization

    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False
    PettingZooVectorization = None

try:
    from mettagrid.mettagrid_env import MettaGridEnv

    METTAGRID_AVAILABLE = True
except ImportError:
    try:
        from mettagrid import MettaGridEnv

        METTAGRID_AVAILABLE = True
    except ImportError:
        METTAGRID_AVAILABLE = False
        MettaGridEnv = None

# Environment presets
ENV_PRESETS = {
    "tiny": {
        "width": 7,
        "height": 7,
        "max_steps": 128,
        "num_agents": 1,
    },
    "small": {
        "width": 11,
        "height": 11,
        "max_steps": 256,
        "num_agents": 1,
    },
    "medium": {
        "width": 15,
        "height": 15,
        "max_steps": 512,
        "num_agents": 2,
    },
    "large": {
        "width": 21,
        "height": 21,
        "max_steps": 1024,
        "num_agents": 4,
    },
    "huge": {
        "width": 31,
        "height": 31,
        "max_steps": 2048,
        "num_agents": 8,
    },
}


def create_env(
    width: int = 11,
    height: int = 11,
    max_steps: int = 512,
    num_agents: int = 1,
    num_resources: int = 3,
    resource_prob: float = 0.01,
    **kwargs,
) -> Any:
    """Create a MettaGrid environment with specified parameters.

    Args:
        width: Width of the grid
        height: Height of the grid
        max_steps: Maximum steps per episode
        num_agents: Number of agents
        num_resources: Number of resource types
        resource_prob: Probability of resource spawn
        **kwargs: Additional environment parameters

    Returns:
        Configured MettaGridEnv instance
    """
    if not METTAGRID_AVAILABLE:
        raise ImportError(
            "mettagrid is required to create environments. Please install it from the mettagrid repository."
        )

    return MettaGridEnv(
        width=width,
        height=height,
        max_steps=max_steps,
        num_agents=num_agents,
        num_resources=num_resources,
        resource_prob=resource_prob,
        **kwargs,
    )


def create_env_from_preset(preset: str, **overrides) -> Any:
    """Create environment from a preset configuration.

    Args:
        preset: Name of preset ("tiny", "small", "medium", "large", "huge")
        **overrides: Override preset parameters

    Returns:
        Configured MettaGridEnv instance
    """
    if preset not in ENV_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(ENV_PRESETS.keys())}")

    config = ENV_PRESETS[preset].copy()
    config.update(overrides)
    return create_env(**config)


def create_vectorized_env(
    num_envs: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
    **env_kwargs,
) -> Any:
    """Create a vectorized environment for parallel training.

    Args:
        num_envs: Total number of environments
        num_workers: Number of worker processes
        device: Device for environment operations
        **env_kwargs: Arguments passed to environment creation

    Returns:
        Vectorized environment instance
    """
    if not PUFFERLIB_AVAILABLE:
        raise ImportError(
            "pufferlib is required for vectorized environments. Please install it with: pip install pufferlib"
        )

    # Create base environment
    env = create_env(**env_kwargs)

    # Create vectorization
    vecenv = PettingZooVectorization(
        env_creator=lambda: create_env(**env_kwargs),
        num_envs=num_envs,
        num_workers=num_workers,
        device=device,
    )

    return vecenv


def create_curriculum_env(
    curriculum_stages: list[dict],
    start_stage: int = 0,
    **base_kwargs,
) -> Any:
    """Create environment with curriculum learning support.

    Args:
        curriculum_stages: List of stage configurations
        start_stage: Initial curriculum stage
        **base_kwargs: Base environment parameters

    Returns:
        Environment with curriculum wrapper
    """
    # For now, just create base env with first stage
    if curriculum_stages:
        stage_config = curriculum_stages[start_stage].copy()
        base_kwargs.update(stage_config)

    return create_env(**base_kwargs)
