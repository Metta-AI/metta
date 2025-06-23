import logging

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agents import create_agent
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


class DistributedMettaAgent(DistributedDataParallel):
    """Wrapper for distributed training of agents."""

    def __init__(self, agent, device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
        agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)

    @property
    def lstm(self):
        """Access LSTM from the wrapped module."""
        return self.module.lstm

    @property
    def total_params(self):
        """Access total_params from the wrapped module."""
        return self.module.total_params

    def l2_reg_loss(self):
        """Access l2_reg_loss from the wrapped module."""
        return self.module.l2_reg_loss()

    def l2_init_loss(self):
        """Access l2_init_loss from the wrapped module."""
        return self.module.l2_init_loss()

    def update_l2_init_weight_copy(self):
        """Access update_l2_init_weight_copy from the wrapped module."""
        self.module.update_l2_init_weight_copy()

    def clip_weights(self):
        """Access clip_weights from the wrapped module."""
        self.module.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01):
        """Access compute_weight_metrics from the wrapped module."""
        return self.module.compute_weight_metrics(delta)


# Keep the old MettaAgent class for backward compatibility, but deprecate it
class MettaAgent(nn.Module):
    """DEPRECATED: Use the new agent classes in metta.agents instead.

    This class is kept for backward compatibility only.
    """

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "MettaAgent is deprecated. Use the new agent classes in metta.agents instead. "
            "For example: from metta.agents import SimpleCNNAgent"
        )
