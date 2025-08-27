import logging
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent

logger = logging.getLogger("policy")


def _parse_weights_metadata(weights: dict[str, Any]) -> tuple[int, int, int, int]:
    """Try to infer num_actions, hidden_size, num_action_args, obs_channels from checkpoint weights."""
    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
        return num_actions, hidden_size, num_action_args, obs_channels
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        logger.warning("Using defaults from config")
        return 9, 512, 10, 22  # Safe defaults


def _init_env() -> SimpleNamespace:
    """Create the runtime env for MettaAgent."""
    obs_shape = [34, 11, 11]
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
        obs_width=obs_shape[1],
        obs_height=obs_shape[2],
        single_action_space=gym.spaces.MultiDiscrete([9, 10]),
        feature_normalizations={},
        global_features=[],
    )


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: Optional[DictConfig] = None) -> MettaAgent:
    """Create or loads a PyTorch policy."""
    # TODO(richard): #dehydration - this is a hack to get the policy to work. We need to fix this.
    raise NotImplementedError("This is a hack to get the policy to work. We need to fix this.")
    # try:
    #     weights = torch.load(path, map_location=device, weights_only=True)
    #     num_actions, hidden_size, num_action_args, obs_channels = _parse_weights_metadata(weights)
    # except Exception as e:
    #     logger.warning(f"Failed to load checkpoint from {path}: {e}")

    # env = _init_env()
    # policy = instantiate(pytorch_cfg, env=env, policy=None)

    # try:
    #     policy.load_state_dict(weights)
    # except Exception as e:
    #     logger.warning(f"Failed to load weights into policy: {e}")
    #     logger.warning("Proceeding with new policy.")

    # logger.info(f"Loaded PyTorch policy config: {pytorch_cfg}")

    # system_cfg = SystemConfig(device=device)

    # return MettaAgent(env, system_cfg, pytorch_cfg, policy=policy)
