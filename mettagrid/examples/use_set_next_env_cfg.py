"""
Example showing how to use the new set_next_env_cfg() method.

This demonstrates how to change environment configuration that will be applied
on the next reset() call.
"""

from typing import Any, Dict


def example_usage_with_curriculum(env):
    """
    Example of how a curriculum might use set_next_env_cfg to prepare
    the next environment configuration before reset.
    """
    # Prepare configuration for the next episode
    next_config: Dict[str, Any] = {
        "num_agents": 4,
        "max_steps": 1000,
        "reward_scale": 1.0,
        # ... other configuration parameters
    }

    # Set the configuration that will be used in the next reset
    env.set_next_env_cfg(next_config)

    # The new configuration will be applied when reset() is called
    obs, info = env.reset()

    # Now the environment is using the new configuration
    return obs, info


def example_adaptive_curriculum(env, performance_score: float):
    """
    Example of an adaptive curriculum that adjusts difficulty based on performance.
    """
    # Determine next configuration based on performance
    if performance_score > 0.8:
        # Increase difficulty
        next_config = {
            "num_agents": 6,
            "max_steps": 800,
            "reward_scale": 0.8,
        }
    elif performance_score < 0.3:
        # Decrease difficulty
        next_config = {
            "num_agents": 2,
            "max_steps": 1200,
            "reward_scale": 1.2,
        }
    else:
        # Keep current difficulty
        next_config = env._env_cfg.copy()

    # Set the next configuration
    env.set_next_env_cfg(next_config)

    # Configuration will be applied on next reset
    return next_config


def example_scheduled_curriculum(env, episode_number: int):
    """
    Example of a scheduled curriculum that changes configuration over time.
    """
    # Define configuration schedule
    if episode_number < 100:
        config = {"difficulty": "easy", "num_agents": 2}
    elif episode_number < 500:
        config = {"difficulty": "medium", "num_agents": 4}
    else:
        config = {"difficulty": "hard", "num_agents": 8}

    # Set the configuration for next reset
    env.set_next_env_cfg(config)

    return config


# Note: The actual implementation ensures that:
# 1. env._next_env_cfg is set when set_next_env_cfg() is called
# 2. During reset(), env._env_cfg = env._next_env_cfg (configuration is applied)
# 3. By default, env._next_env_cfg = env._env_cfg (no change if not explicitly set)
