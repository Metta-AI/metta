from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class SimpleNPCPolicy(Policy):
    """Non-trainable NPC baseline that emits a safe, deterministic action."""

    def __init__(self, policy_env_info: PolicyEnvInterface, *, default_action: str = "noop") -> None:
        super().__init__(policy_env_info)
        self._device = torch.device("cpu")
        self._default_action_name = default_action
        self._default_action_id = self._resolve_action_id(default_action)

    def _resolve_action_id(self, target: str) -> Optional[int]:
        for idx, action in enumerate(self._policy_env_info.actions.actions()):
            if getattr(action, "name", None) == target:
                return idx
        return None

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device) -> None:
        self._policy_env_info = policy_env_info
        self._default_action_id = self._resolve_action_id(self._default_action_name)
        self._device = torch.device(device)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        base = super().get_agent_experience_spec()
        act_dtype = torch.int64
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        merged = dict(base.items())
        merged.update(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            act_log_prob=scalar_f32,
            entropy=scalar_f32,
            values=scalar_f32,
        )
        return Composite(merged)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        batch = td.batch_size.numel()
        target_device = td.device

        if action is not None:
            actions = action.to(device=target_device, dtype=torch.int64).view(-1)
        else:
            action_id = self._default_action_id if self._default_action_id is not None else 0
            actions = torch.full((batch,), action_id, device=target_device, dtype=torch.int64)

        td.set("actions", actions)
        td.set("act_log_prob", torch.zeros(batch, device=target_device, dtype=torch.float32))
        td.set("entropy", torch.zeros(batch, device=target_device, dtype=torch.float32))
        td.set("values", torch.zeros(batch, device=target_device, dtype=torch.float32))
        return td

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device

    def reset_memory(self) -> None:  # noqa: D401
        return None
