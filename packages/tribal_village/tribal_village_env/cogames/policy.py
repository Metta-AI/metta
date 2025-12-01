"""PufferLib policy wrapper for the Tribal Village environment."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mettagrid.simulator import AgentObservation

from .policy_base import DefaultPufferPolicy


@dataclass
class TribalPolicyEnvInfo:
    """Minimal environment metadata required by MultiAgentPolicy."""

    observation_space: spaces.Box
    action_space: spaces.Discrete
    num_agents: int

    @property
    def action_names(self) -> list[str]:
        return [f"action_{idx}" for idx in range(self.action_space.n)]

    @property
    def actions(self) -> list[SimpleNamespace]:
        """Adapter expected by MultiAgentPolicy/PolicyEnvInterface."""

        return [SimpleNamespace(name=name) for name in self.action_names]

    def as_shim_env(self) -> SimpleNamespace:
        """Shape-compatible shim used by PufferLib's default model."""

        shim = SimpleNamespace(
            single_observation_space=self.observation_space,
            single_action_space=self.action_space,
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
        )
        shim.env = shim
        return shim


class TribalVillagePufferPolicy(DefaultPufferPolicy):
    """Trainable policy using PufferLib's default model for Tribal Village."""

    short_names = ["tribal", "tribal_default", "tribal_puffer"]

    def __init__(
        self,
        policy_env_info: TribalPolicyEnvInfo,
        *,
        hidden_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        info_shape = policy_env_info.observation_space.shape

        def _shim(info: TribalPolicyEnvInfo) -> Any:
            return info.as_shim_env()

        def _obs_adapter(obs: Union[AgentObservation, np.ndarray, Sequence[Any]]) -> np.ndarray:
            if isinstance(obs, AgentObservation):
                return np.zeros(info_shape, dtype=np.float32)
            return np.asarray(obs, dtype=np.float32)

        super().__init__(
            policy_env_info=policy_env_info,
            hidden_size=hidden_size,
            device=device,
            shim_factory=_shim,
            obs_adapter=_obs_adapter,
        )
