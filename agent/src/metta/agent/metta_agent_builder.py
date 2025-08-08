import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf, DictConfig

from metta.agent.metta_agent import MettaAgent, ComponentPolicy
from metta.agent.pytorch.agent_mapper import agent_classes

logger = logging.getLogger("metta_agent_builder")


class MettaAgentBuilder:
    """Builder class for constructing MettaAgent instances with validated configurations."""

    def __init__(self, env: 'MettaGridEnv', env_cfg, agent_cfg):
        """
        Initialize the MettaAgentBuilder with environment and configuration.

        Args:
            env (MettaGridEnv): The environment providing observation and action spaces.
            cfg (DictConfig): Configuration for the agent, expected to contain an 'agent' section.
        """
        self.env = env
        self.cfg = env_cfg
        self.agent_cfg = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
        self.obs_space = self._create_observation_space()


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
        # Check if agent_cfg specifies a pytorch agent type
        if "agent_type" in self.agent_cfg and self.agent_cfg.agent_type in agent_classes:
            AgentClass = agent_classes[self.agent_cfg.agent_type]
            agent = AgentClass(env=self.env)
            logger.info(f"Using Pytorch Policy: {agent} (type: {self.agent_cfg.agent_type})")
            return MettaAgent(
                obs_space=self.obs_space,
                obs_width=self.env.obs_width,
                obs_height=self.env.obs_height,
                action_space=self.env.single_action_space,
                feature_normalizations=self.env.feature_normalizations,
                device="cpu",
                cfg=self.agent_cfg,
                policy=agent
            )

        try:
            policy = policy or ComponentPolicy()


            agent = MettaAgent(
                obs_space=self.obs_space,
                obs_width=self.env.obs_width,
                obs_height=self.env.obs_height,
                action_space=self.env.single_action_space,
                feature_normalizations=self.env.feature_normalizations,
                device="cpu",
                cfg=self.agent_cfg,
                policy=policy
            )

            logger.info(f"Successfully built MettaAgent with policy: {type(policy).__name__}")
            return agent

        except Exception as e:
            logger.error(f"Failed to build MettaAgent: {e}")
            raise RuntimeError(f"Agent construction failed: {e}")
