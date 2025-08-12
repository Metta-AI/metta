import logging
from typing import TYPE_CHECKING, Optional

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.agent.component_policy import ComponentPolicy
from metta.agent.metta_agent import MettaAgent
from metta.agent.pytorch.agent_mapper import agent_classes
from metta.rl.system_config import SystemConfig

if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent_builder")


class MettaAgentBuilder:
    """Builder for constructing MettaAgent instances with validated configurations."""

    def __init__(self, env: "MettaGridEnv", system_cfg: SystemConfig, agent_cfg: DictConfig):
        """
        Args:
            env (MettaGridEnv): Environment with observation and action spaces.
            system_cfg (SystemConfig): System configuration.
            agent_cfg (DictConfig): Agent configuration, expected to contain an 'agent' section.
        """
        self.env = env
        self.system_cfg = system_cfg
        self.agent_cfg = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
        self.obs_space = self._create_observation_space()

    def _create_observation_space(self) -> gym.spaces.Dict:
        """Combine grid observations and global variables into an observation space."""
        return gym.spaces.Dict(
            {
                "grid_obs": self.env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.int32),
            }
        )

    def build(self, policy: Optional[ComponentPolicy] = None) -> MettaAgent:
        """
        Build a MettaAgent instance with either a specified policy or a default based on configuration.
        """
        if self.agent_cfg.get("agent_type") in agent_classes:
            AgentClass = agent_classes[self.agent_cfg.agent_type]
            policy = AgentClass(env=self.env)
            logger.info(f"Using PyTorch Policy: {policy} (type: {self.agent_cfg.agent_type})")
        else:
            # ComponentPolicy needs the config to build components, just like old MettaAgent
            policy = policy or ComponentPolicy(
                obs_space=self.obs_space,
                obs_width=self.env.obs_width,
                obs_height=self.env.obs_height,
                action_space=self.env.single_action_space,
                feature_normalizations=self.env.feature_normalizations,
                device=self.system_cfg.device,
                cfg=self.agent_cfg,
            )
            logger.info(f"Using ComponentPolicy: {type(policy).__name__}")

        agent = self._create_agent(policy=policy)
        logger.info(f"Successfully built MettaAgent with policy: {type(policy).__name__}")
        return agent

    def _create_agent(self, policy: ComponentPolicy) -> MettaAgent:
        """Helper to construct a MettaAgent with the given policy."""
        return MettaAgent(
            obs_space=self.obs_space,
            obs_width=self.env.obs_width,
            obs_height=self.env.obs_height,
            action_space=self.env.single_action_space,
            feature_normalizations=self.env.feature_normalizations,
            device=self.system_cfg.device,
            cfg=self.agent_cfg,
            policy=policy,
        )
