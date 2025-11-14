from __future__ import annotations

from typing import Any

import torch

from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class CheckpointPolicy(MultiAgentPolicy):
    """MultiAgentPolicy adapter that instantiates policies from stored checkpoints."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        policy_uri: str,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> None:
        super().__init__(policy_env_info)
        torch_device = torch.device(device)

        artifact = CheckpointManager.load_artifact_from_uri(policy_uri)
        policy = artifact.instantiate(policy_env_info, device=torch_device, strict=strict)
        policy = policy.to(torch_device)
        policy.eval()
        self._policy = policy
        self._device = torch_device

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step_batch(self, raw_observations, raw_actions) -> None:  # type: ignore[override]
        if hasattr(self._policy, "step_batch"):
            self._policy.step_batch(raw_observations, raw_actions)
        else:
            super().step_batch(raw_observations, raw_actions)

    def to(self, device: torch.device) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        self._device = device
        if hasattr(self._policy, "to"):
            self._policy = self._policy.to(device)
        return self

    def eval(self) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        if hasattr(self._policy, "eval"):
            self._policy.eval()
        return self

    def train(self, mode: bool = True) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        if hasattr(self._policy, "train"):
            self._policy.train(mode)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not callable(self._policy):
            raise TypeError(f"Underlying policy {type(self._policy).__name__} is not callable")
        return self._policy(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Provide nn.Module-like API so direct forward() calls also work.
        return self.__call__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self._policy, item)
