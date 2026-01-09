from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from pydantic import Field, model_validator
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite

from metta.agent.policy import Policy
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri


class LossProfileConfig(Config):
    losses: list[str] = Field(default_factory=list)


class PolicySlotConfig(Config):
    id: str = Field(description="Unique slot identifier")
    policy_uri: Optional[str] = Field(default=None, description="Checkpoint URI for neural policies")
    class_path: Optional[str] = Field(default=None, description="Import path for scripted policies")
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    architecture_spec: Optional[str] = Field(
        default=None,
        description="Optional policy architecture spec for checkpoint bundles when class_path is used.",
    )
    trainable: bool = Field(default=True, description="Whether gradients should flow for this slot")
    loss_profile: Optional[str] = Field(default=None, description="Optional loss profile name for this slot")
    use_trainer_policy: bool = Field(default=False, description="Reuse trainer-provided policy instance")

    @model_validator(mode="after")
    def validate_loader(self) -> "PolicySlotConfig":
        if not self.use_trainer_policy and not (self.policy_uri or self.class_path):
            raise ValueError("policy_uri or class_path must be set unless use_trainer_policy=True")
        if self.use_trainer_policy and (self.policy_uri or self.class_path):
            raise ValueError("use_trainer_policy=True is mutually exclusive with policy_uri/class_path")
        return self


class SlotRegistry:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Policy] = {}

    def _cache_key(self, slot: PolicySlotConfig, device: torch.device) -> Tuple[str, str]:
        key_dict = {
            "uri": slot.policy_uri,
            "class_path": slot.class_path,
            "kwargs": slot.policy_kwargs,
            "device": str(device),
        }
        return (slot.id, json.dumps(key_dict, sort_keys=True))

    def get(self, slot: PolicySlotConfig, policy_env_info: PolicyEnvInterface, device: torch.device) -> Policy:
        if slot.use_trainer_policy:
            raise ValueError("use_trainer_policy slots must be supplied externally, not loaded via registry")

        key = self._cache_key(slot, device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if slot.policy_uri:
            policy_spec = policy_spec_from_uri(slot.policy_uri, device=str(device))
            policy = initialize_or_load_policy(policy_env_info, policy_spec, device_override=str(device))
        elif slot.class_path:
            PolicyCls = load_symbol(slot.class_path)
            policy = PolicyCls(policy_env_info, **slot.policy_kwargs)
            if not isinstance(policy, Policy):
                raise TypeError(f"Loaded policy from {slot.class_path} is not a metta.agent.policy.Policy instance")
        else:
            raise ValueError("Slot must provide policy_uri or class_path to load a policy")

        policy = policy.to(device)
        policy.initialize_to_environment(policy_env_info, device)
        self._cache[key] = policy
        return policy


def _merge_policy_specs(specs: Iterable[Composite]) -> Composite:
    merged: dict[str, Any] = {}
    for spec in specs:
        for key in spec.keys(include_nested=True, leaves_only=True):
            value = spec.get(key)
            existing = merged.get(key)
            if existing is not None:
                if getattr(existing, "shape", None) != getattr(value, "shape", None) or getattr(
                    existing, "dtype", None
                ) != getattr(value, "dtype", None):
                    raise ValueError(
                        f"Slot policies disagree on spec for '{key}': "
                        f"{getattr(existing, 'shape', None)}/{getattr(existing, 'dtype', None)} vs "
                        f"{getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}"
                    )
                continue
            merged[key] = value
    return Composite(merged)


def apply_slot_routing(
    td: TensorDict,
    *,
    teacher_mask: torch.Tensor,
    student_mask: torch.Tensor,
    ppo_mask: torch.Tensor,
    teacher_slot_id: str | None,
    student_slot_id: str | None,
    slot_lookup: dict[str, int],
) -> None:
    if not teacher_slot_id or not student_slot_id:
        return
    if "slot_id" not in td.keys() or not slot_lookup:
        return

    teacher_slot_idx = slot_lookup.get(teacher_slot_id)
    student_slot_idx = slot_lookup.get(student_slot_id)
    if teacher_slot_idx is None or student_slot_idx is None:
        raise RuntimeError(
            f"Slot routing requires slot ids '{student_slot_id}' and '{teacher_slot_id}' "
            f"to exist in trainer.policy_slots."
        )

    slot_ids = td.get("slot_id").to(device=td.device)
    flat_slots = slot_ids.reshape(-1)
    flat_slots[teacher_mask] = teacher_slot_idx
    flat_slots[student_mask | ppo_mask] = student_slot_idx
    td["slot_id"] = flat_slots.reshape(slot_ids.shape)


class SlotControllerPolicy(Policy):
    """Multi-policy controller that routes agent observations to different slot policies.

    Performance considerations:
    - Each unique slot triggers a separate policy forward pass
    - Best performance with few unique slots (1-4)
    - Agents using the same slot are batched together
    - Consider using agent_slot_map for static routing (avoids per-step slot_id computation)
    - Non-trainable slots should use lightweight policies (e.g., SimpleNPCPolicy)

    Memory usage:
    - Each slot policy maintains its own parameters and buffers
    - Shared slots reduce memory overhead vs per-agent policies
    - Use slot caching (via SlotRegistry) to avoid duplicate policy loads
    """

    def __init__(
        self,
        slot_lookup: Dict[str, int],
        slots: list[Any],
        slot_policies: Dict[int, Policy],
        policy_env_info,
        controller_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        agent_slot_map: torch.Tensor | None = None,
    ) -> None:
        super().__init__(policy_env_info)  # type: ignore[arg-type]

        self._slot_lookup = slot_lookup
        self._slots = slots
        self._slot_policies = slot_policies
        self._policy_env_info = policy_env_info
        chosen = controller_device if controller_device is not None else device
        if chosen is not None:
            self._device = torch.device(chosen)
        else:
            first_policy = next(iter(slot_policies.values()), None)
            policy_device = getattr(first_policy, "device", None) if first_policy is not None else None
            self._device = torch.device(policy_device) if policy_device is not None else torch.device("cpu")
        if agent_slot_map is not None:
            self.register_buffer("_agent_slot_map", agent_slot_map)
        else:
            self._agent_slot_map = None
        self._slot_specs = {idx: policy.get_agent_experience_spec() for idx, policy in slot_policies.items()}
        specs_for_merge: list[Composite] = []
        for idx, spec in self._slot_specs.items():
            slot = slots[idx] if idx < len(slots) else None
            if slot is None or getattr(slot, "trainable", True):
                specs_for_merge.append(spec)
        if not specs_for_merge:
            specs_for_merge = list(self._slot_specs.values())
        self._merged_spec = _merge_policy_specs(specs_for_merge)

        for idx, policy in slot_policies.items():
            if isinstance(policy, nn.Module):
                self.add_module(f"slot_{idx}", policy)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return self._merged_spec

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.initialize_to_environment(policy_env_info, device)
        self._device = torch.device(device)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        if "slot_id" not in td.keys():
            slot_map = self._agent_slot_map
            if slot_map is None:
                raise RuntimeError("slot_id missing and agent_slot_map is not configured.")
            num_agents = int(slot_map.numel())
            if num_agents == 0:
                raise RuntimeError("agent_slot_map cannot be empty for slot-aware policy routing.")
            num_envs = td.batch_size.numel() // num_agents
            if slot_map.device != td.device:
                slot_map = slot_map.to(device=td.device)
            reshaped_map = slot_map.view(1, num_agents).expand(num_envs, num_agents)
            td.set("slot_id", reshaped_map.reshape(td.batch_size))

        slot_ids = td.get("slot_id")

        unique_ids: Iterable[int] = torch.unique(slot_ids).tolist()
        for b_id in unique_ids:
            mask = slot_ids == b_id

            sub_td = td[mask].clone()
            policy = self._slot_policies.get(int(b_id))
            out_td = policy.forward(sub_td, action=None if action is None else action[mask])

            for key in out_td.keys(include_nested=True, leaves_only=True):
                value = out_td.get(key)
                if not isinstance(value, torch.Tensor):
                    continue
                if key not in td.keys():
                    full_shape = td.batch_size + value.shape[1:]
                    td.set(key, value.new_zeros(full_shape))
                td.set_at_(key, value, mask)

        return td

    @property
    def slot_policies(self) -> Dict[int, Policy]:
        return self._slot_policies

    @property
    def slot_specs(self) -> Dict[int, Composite]:
        return self._slot_specs

    def reset_memory(self) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.reset_memory()

    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        for idx, policy in self._slot_policies.items():
            if not getattr(self._slots[idx], "trainable", True):
                policy.eval()
        return self

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device
