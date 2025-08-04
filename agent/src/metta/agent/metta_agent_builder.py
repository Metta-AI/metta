import logging
from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple

import gymnasium as gym
from omegaconf import OmegaConf, DictConfig
from torch import nn
import torch

from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.instantiate import instantiate
from metta.agent.policy_base import PolicyBase
from metta.agent.policy_state import PolicyState
from metta.agent.metta_agent import MettaAgent, ComponentPolicy

import numpy as np



logger = logging.getLogger("metta_agent_builder")


def make_policy(env: "MettaGridEnv", cfg: DictConfig) -> MettaAgent:

    """Factory function to create MettaAgent from environment and config."""
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
    })

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    logger.info(f"Agent Config: {OmegaConf.create(agent_cfg)}")

    logger.info(f"Feature Normalizations: {env.feature_normalizations}")

    builder = MettaAgentBuilder(
        agent_cfg,
    )

    return builder.build(env, obs_space)


class MettaAgentBuilder:
    """Simplified builder for MettaAgent instances."""

    def __init__(self, cfg):
        self.cfg = OmegaConf.create(cfg)


    def build(self, env, obs_space) -> MettaAgent:
        """Build the final MettaAgent instance."""

        policy = ComponentPolicy()
        try:
            agent = MettaAgent(
                    obs_space=obs_space,
                    obs_width=env.obs_width,
                    obs_height=env.obs_height,
                    action_space=env.single_action_space,
                    feature_normalizations=env.feature_normalizations,
                    device="cpu",
                    cfg=self.cfg,
                    policy=policy
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to build MettaAgent: {e}")
            raise


