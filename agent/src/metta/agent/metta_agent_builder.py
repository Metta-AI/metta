import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf, DictConfig

from metta.agent.metta_agent import MettaAgent, ComponentPolicy
from metta.agent.pytorch.agent_mapper import agent_classes

logger = logging.getLogger("metta_agent_builder")


class MettaAgentBuilder:
    """Builder for constructing MettaAgent instances with validated configurations."""

    def __init__(self, env: "MettaGridEnv", env_cfg: DictConfig | None, agent_cfg: DictConfig):
        """
        Args:
            env (MettaGridEnv): Environment with observation and action spaces.
            env_cfg (DictConfig): Environment configuration.
            agent_cfg (DictConfig): Agent configuration, expected to contain an 'agent' section.
        """
        self.env = env
        self.cfg = env_cfg
        self.agent_cfg = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
        self.obs_space = self._create_observation_space()

    def _create_observation_space(self) -> gym.spaces.Dict:
        """Combine grid observations and global variables into an observation space."""
        return gym.spaces.Dict({
            "grid_obs": self.env.single_observation_space,
            "global_vars": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.int32
            ),
        })

    def build(self, policy: Optional[ComponentPolicy] = None) -> MettaAgent:
        """
        Build a MettaAgent instance.

        Args:
            policy: Optional policy to use. Defaults to ComponentPolicy if None.

        Returns:
            MettaAgent: Constructed agent instance.
        """
        # If a PyTorch-based agent type is specified, use it
        if self.agent_cfg.get("agent_type") in agent_classes:
            AgentClass = agent_classes[self.agent_cfg.agent_type]
            pytorch_agent = AgentClass(env=self.env)
            logger.info(f"Using PyTorch Policy: {pytorch_agent} (type: {self.agent_cfg.agent_type})")
            return self._create_agent(policy=pytorch_agent)

        # Otherwise, use provided or default ComponentPolicy
        try:
            policy = policy or ComponentPolicy()
            agent = self._create_agent(policy=policy)
            logger.info(f"Successfully built MettaAgent with policy: {type(policy).__name__}")
            return agent
        except Exception as e:
            logger.exception("Failed to build MettaAgent")
            raise RuntimeError(f"Agent construction failed: {e}") from e


    def _create_agent(self, policy: ComponentPolicy) -> MettaAgent:
        """Helper to construct a MettaAgent with the given policy."""
        return MettaAgent(
            obs_space=self.obs_space,
            obs_width=self.env.obs_width,
            obs_height=self.env.obs_height,
            action_space=self.env.single_action_space,
            feature_normalizations=self.env.feature_normalizations,
            device="cpu",
            cfg=self.agent_cfg,
            policy=policy,
        )
