"""Binding-aware policy controller that routes per-agent batches to the correct policy."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite

from metta.agent.policy import Policy


class BindingControllerPolicy(Policy):
    """A thin wrapper that fans out a rollout TensorDict to per-binding policies and merges outputs."""

    def __init__(
        self,
        binding_lookup: Dict[str, int],
        bindings: list[Any],
        binding_policies: Dict[int, Policy],
        policy_env_info,
        device: torch.device,
    ) -> None:
        # Use the env info from trainer policy; architecture not needed here
        super().__init__(policy_env_info)  # type: ignore[arg-type]
        self._binding_lookup = binding_lookup
        self._bindings = bindings
        self._binding_policies = binding_policies
        self._policy_env_info = policy_env_info
        self._device = torch.device(device)

        # Register trainable sub-policies so optimizer sees their parameters
        for idx, policy in binding_policies.items():
            if isinstance(policy, nn.Module):
                self.add_module(f"binding_{idx}", policy)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        # Assume all policies share the same spec; use the first one
        first_policy = next(iter(self._binding_policies.values()))
        return first_policy.get_agent_experience_spec()

    def initialize_to_environment(self, policy_env_info, device: torch.device) -> None:  # noqa: D401
        for policy in self._binding_policies.values():
            policy.initialize_to_environment(policy_env_info, device)
        self._device = torch.device(device)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        if "binding_id" not in td.keys():
            # Fallback: single policy scenario
            first_policy = next(iter(self._binding_policies.values()))
            return first_policy.forward(td, action=action)

        binding_ids = td.get("binding_id")
        if not isinstance(binding_ids, torch.Tensor):
            raise RuntimeError("binding_id must be a tensor")

        unique_ids: Iterable[int] = torch.unique(binding_ids).tolist()
        for b_id in unique_ids:
            mask = binding_ids == b_id
            if not torch.any(mask):
                continue

            sub_td = td[mask].clone()
            policy = self._binding_policies.get(int(b_id))
            if policy is None:
                raise RuntimeError(f"No policy registered for binding id {int(b_id)}")

            out_td = policy.forward(sub_td, action=None if action is None else action[mask])
            td[mask] = out_td

        return td

    def reset_memory(self) -> None:  # noqa: D401
        for policy in self._binding_policies.values():
            policy.reset_memory()

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device
