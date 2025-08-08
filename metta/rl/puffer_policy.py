import logging
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.common.util.instantiate import instantiate

logger = logging.getLogger("policy")


def load_pytorch_policy(path: str, device: str = "cpu", cfg = None, pytorch_cfg: DictConfig | None = None) -> "PytorchAgent":
    """Load a PyTorch policy from checkpoint and wrap it in PytorchAgent.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy with _target_ field

    Returns:
        PytorchAgent wrapping the loaded policy
    """

    try:
        weights = torch.load(path, map_location=device, weights_only=True)

        weights = torch.load(path, map_location=device, weights_only=True)

        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        logger.warning("Using defaults from config")
        num_actions = 9
        hidden_size = 512
        num_action_args = 10
        obs_channels = 22  # Updated default to 22 channels

    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
    )

    # Use common instantiate function
    if pytorch_cfg is None:
        # Default to Recurrent policy if no config provided
        from metta.agent.external.example import Policy, Recurrent

        # Create the Policy first
        policy = Policy(
            env=env,
            cnn_channels=128,
            hidden_size=hidden_size,
        )

        # Then wrap it in Recurrent
        policy = Recurrent(
            env=env,
            policy=policy,
            input_size=512,
            hidden_size=hidden_size,
        )
    else:
        # Use the common instantiate utility
        policy = instantiate(pytorch_cfg, env=env, policy=None)
    try:
        policy.load_state_dict(weights)

    except Exception as e:
        logger.warning(f"Failed automatic load from weights: {e}")
        logger.warning("Using randomly initialized weights")


    from metta.agent.metta_agent_builder import MettaAgentBuilder
    import gymnasium as gym
    import numpy as np

    obs_shape = [34, 11, 11]
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
        obs_width=obs_shape[1],
        obs_height=obs_shape[2],
        single_action_space=gym.spaces.MultiDiscrete([9, 10]),
        feature_normalizations={},
        global_features=[],
    )

    builder = MettaAgentBuilder(env, cfg)

    metta_agent = builder.build(policy)
    return metta_agent



class PytorchPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, x):
        return self.policy(x)
