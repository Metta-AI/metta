import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf, DictConfig

from metta.agent.metta_agent import MettaAgent, ComponentPolicy

logger = logging.getLogger("metta_agent_builder")


class MettaAgentBuilder:
    """Builder class for constructing MettaAgent instances with validated configurations."""

    def __init__(self, env: 'MettaGridEnv', cfg: DictConfig):
        """
        Initialize the MettaAgentBuilder with environment and configuration.

        Args:
            env (MettaGridEnv): The environment providing observation and action spaces.
            cfg (DictConfig): Configuration for the agent, expected to contain an 'agent' section.
        """
        self.env = env
        self.cfg = self._parse_config(cfg)
        self.obs_space = self._create_observation_space()

    def _parse_config(self, cfg: DictConfig) -> DictConfig:
        """
        Parse and validate the configuration.

        Args:
            cfg (DictConfig): Input configuration.

        Returns:
            DictConfig: Parsed and resolved agent configuration.

        Raises:
            ValueError: If the configuration is invalid or missing required sections.
        """
        if not hasattr(cfg, "agent"):
            logger.error("Configuration missing 'agent' section")
            raise ValueError("Configuration must contain 'agent' section")

        try:
            agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=True))
            logger.info(f"Agent Config: {agent_cfg}")
            return agent_cfg

        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            raise ValueError(f"Invalid configuration format: {e}")

    def _create_observation_space(self) -> gym.spaces.Dict:
        """
        Create the observation space for the agent.

        Returns:
            gym.spaces.Dict: The observation space combining grid observations and global variables.
        """
        return gym.spaces.Dict({
            "grid_obs": self.env.single_observation_space,
            "global_vars": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=[0],
                dtype=np.int32
            )
        })

    def build(self, policy: Optional[ComponentPolicy] = None) -> MettaAgent:
        """
        Build the MettaAgent instance with the specified or default policy.

        Args:
            policy (Optional[ComponentPolicy]): The policy to use; defaults to ComponentPolicy if None.

        Returns:
            MettaAgent: The constructed agent instance.

        Raises:
            RuntimeError: If agent construction fails.
        """
        try:
            policy = policy or ComponentPolicy()

            agent = MettaAgent(
                obs_space=self.obs_space,
                obs_width=self.env.obs_width,
                obs_height=self.env.obs_height,
                action_space=self.env.single_action_space,
                feature_normalizations=self.env.feature_normalizations,
                device="cpu",
                cfg=self.cfg,
                policy=policy
            )

            logger.info(f"Successfully built MettaAgent with policy: {type(policy).__name__}")
            return agent

        except Exception as e:
            logger.error(f"Failed to build MettaAgent: {e}")
            raise RuntimeError(f"Agent construction failed: {e}")
