"""Slot-aware policy controller that routes per-agent batches to the correct policy."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite

from metta.agent.policy import Policy


class SlotControllerPolicy(Policy):
    """A thin wrapper that fans out a rollout TensorDict to per-slot policies and merges outputs."""

    def __init__(
        self,
        slot_lookup: Dict[str, int],
        slots: list[Any],
        slot_policies: Dict[int, Policy],
        policy_env_info,
        agent_slot_map: torch.Tensor | None = None,
    ) -> None:
        # Use the env info from trainer policy; architecture not needed here
        super().__init__(policy_env_info)  # type: ignore[arg-type]
        self._slot_lookup = slot_lookup
        self._slots = slots
        self._slot_policies = slot_policies
        self._policy_env_info = policy_env_info
        if agent_slot_map is not None:
            self.register_buffer("_agent_slot_map", agent_slot_map)
        else:
            self._agent_slot_map = None

        # Register trainable sub-policies so optimizer sees their parameters
        for idx, policy in slot_policies.items():
            if isinstance(policy, nn.Module):
                self.add_module(f"slot_{idx}", policy)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return next(iter(self._slot_policies.values())).get_agent_experience_spec()

    def initialize_to_environment(self, policy_env_info, device: torch.device) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.initialize_to_environment(policy_env_info, device)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        if "slot_id" not in td.keys():
            if self._agent_slot_map is None:
                raise RuntimeError("slot_id missing and no agent_slot_map provided for slot-aware policy routing")
            num_agents = self._agent_slot_map.numel()
            batch = td.batch_size.numel()
            if batch % num_agents != 0:
                raise RuntimeError(
                    f"slot-aware routing requires batch size ({batch}) to be divisible by num_agents ({num_agents})"
                )
            num_envs = batch // num_agents
            slot_map = self._agent_slot_map
            if slot_map.device != td.device:
                slot_map = slot_map.to(device=td.device)
            td.set("slot_id", slot_map.repeat(num_envs))

        slot_ids = td.get("slot_id")
        assert isinstance(slot_ids, torch.Tensor), "slot_id must be a tensor"

        unique_ids: Iterable[int] = torch.unique(slot_ids).tolist()
        for b_id in unique_ids:
            mask = slot_ids == b_id

            sub_td = td[mask].clone()
            policy = self._slot_policies.get(int(b_id))
            assert policy is not None, f"No policy registered for slot id {int(b_id)}"

            out_td = policy.forward(sub_td, action=None if action is None else action[mask])

            for key in ("actions", "act_log_prob", "entropy", "values", "h_values", "full_log_probs", "logits"):
                if key not in out_td.keys():
                    continue
                value = out_td.get(key)
                if key not in td.keys():
                    full_shape = td.batch_size + value.shape[1:]
                    zeroed = value.new_zeros(full_shape)
                    td.set(key, zeroed)
                td.set_at_(key, value, mask)

        return td

    def reset_memory(self) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.reset_memory()

    @property
    def device(self) -> torch.device:  # noqa: D401
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")
