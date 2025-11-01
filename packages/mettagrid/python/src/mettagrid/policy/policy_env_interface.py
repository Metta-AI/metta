"""Policy environment interface for providing environment information to policies."""

from dataclasses import dataclass

import gymnasium as gym

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig, MettaGridConfig


@dataclass
class PolicyEnvInterface:
    """Interface providing environment information needed by policies.

    This class encapsulates the environment configuration details that policies
    need to initialize their networks, such as observation dimensions, action spaces,
    and agent counts.
    """

    obs_features: list[ObservationFeatureSpec]
    actions: ActionsConfig
    num_agents: int
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete
    obs_width: int
    obs_height: int

    @staticmethod
    def from_mg_cfg(mg_cfg: MettaGridConfig) -> "PolicyEnvInterface":
        """Create PolicyEnvInterface from MettaGridConfig.

        Args:
            mg_cfg: The MettaGrid configuration

        Returns:
            A PolicyEnvInterface instance with environment information
        """
        return PolicyEnvInterface(
            obs_features=mg_cfg.id_map().features(),
            actions=mg_cfg.game.actions,
            num_agents=mg_cfg.game.num_agents,
            observation_space=gym.spaces.Box(0, 255, (mg_cfg.game.obs.num_tokens, mg_cfg.game.obs.token_dim)),
            action_space=gym.spaces.Discrete(len(mg_cfg.game.actions.actions())),
            obs_width=mg_cfg.game.obs.width,
            obs_height=mg_cfg.game.obs.height,
        )
