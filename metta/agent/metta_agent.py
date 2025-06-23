import logging

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, ListConfig
from torch import nn

from metta.agent import create_agent
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig):
    """Create a policy using the new agent architecture."""
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Get agent name from config
    if hasattr(cfg, "agent") and hasattr(cfg.agent, "name"):
        agent_name = cfg.agent.name
    elif hasattr(cfg, "agent") and hasattr(cfg.agent, "_target_"):
        # Try to extract agent name from old-style config
        target = cfg.agent._target_
        if "simple" in target.lower():
            agent_name = "simple_cnn"
        elif "large" in target.lower():
            agent_name = "large_cnn"
        elif "attention" in target.lower():
            agent_name = "attention"
        else:
            agent_name = "simple_cnn"  # default
    else:
        agent_name = "simple_cnn"  # default

    # Create the agent using the factory
    agent = create_agent(
        agent_name=agent_name,
        obs_space=obs_space,
        action_space=env.single_action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device=cfg.device,
    )

    return agent


# Keep the old MettaAgent class for backward compatibility, but deprecate it
class MettaAgent(nn.Module):
    """DEPRECATED: Use the new agent classes in metta.agent instead.

    This class is kept for backward compatibility only.
    """

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "MettaAgent is deprecated. Use the new agent classes in metta.agent instead. "
            "For example: from metta.agent import SimpleCNNAgent"
        )
