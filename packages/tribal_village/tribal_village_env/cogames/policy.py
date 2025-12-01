"""PufferLib policy wrapper for the Tribal Village environment."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from gymnasium import spaces

from cogames.policy.pufferlib_policy import PufferlibCogsPolicy


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


class TribalVillagePufferPolicy(PufferlibCogsPolicy):
    """Trainable policy using PufferLib's default model for Tribal Village."""

    short_names = ["tribal", "tribal_default", "tribal_puffer"]

    def __init__(
        self,
        policy_env_info: TribalPolicyEnvInfo,
        *,
        hidden_size: int = 256,
        device: Optional[object] = None,
    ) -> None:
        super().__init__(policy_env_info, hidden_size=hidden_size, device=device)
