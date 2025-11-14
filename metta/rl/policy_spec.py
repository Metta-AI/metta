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
        display_name: str | None = None,
    ) -> None:
        super().__init__(policy_env_info)
        torch_device = torch.device(device)

        strict_value = strict
        if isinstance(strict_value, str):
            strict_value = strict_value.lower() not in {"0", "false", "no"}

        artifact = CheckpointManager.load_artifact_from_uri(policy_uri)
        policy = artifact.instantiate(policy_env_info, device=torch_device, strict=strict_value)
        policy = policy.to(torch_device)
        policy.eval()
        self._policy = policy
        self._device = torch_device
        self._display_name = display_name or policy_uri

    def _call_policy_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._policy, name, None)
        if callable(method):
            return method(*args, **kwargs)
        return None

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def reset(self) -> None:
        self._call_policy_method("reset")

    def step_batch(self, raw_observations, raw_actions) -> None:  # type: ignore[override]
        if self._call_policy_method("step_batch", raw_observations, raw_actions) is None:
            super().step_batch(raw_observations, raw_actions)

    def to(self, device: torch.device) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        self._device = device
        new_policy = self._call_policy_method("to", device)
        if new_policy is not None:
            self._policy = new_policy
        return self

    def eval(self) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        self._call_policy_method("eval")
        return self

    def train(self, mode: bool = True) -> "CheckpointPolicy":  # pragma: no cover - convenience passthrough
        self._call_policy_method("train", mode)
        return self

    @property
    def display_name(self) -> str:
        return self._display_name

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
