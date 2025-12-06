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
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol


class LossProfileConfig(Config):
    losses: list[str] = Field(default_factory=list)


class PolicySlotConfig(Config):
    id: str = Field(description="Unique slot identifier")
    policy_uri: Optional[str] = Field(default=None, description="Checkpoint URI for neural policies")
    class_path: Optional[str] = Field(default=None, description="Import path for scripted policies")
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
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

    def _cache_key(self, slot: PolicySlotConfig) -> Tuple[str, str]:
        key_dict = {"uri": slot.policy_uri, "class_path": slot.class_path, "kwargs": slot.policy_kwargs}
        return (slot.id, json.dumps(key_dict, sort_keys=True))

    def get(self, slot: PolicySlotConfig, policy_env_info: PolicyEnvInterface, device: torch.device) -> Policy:
        if slot.use_trainer_policy:
            raise ValueError("use_trainer_policy slots must be supplied externally, not loaded via registry")

        key = self._cache_key(slot)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from metta.rl.checkpoint_manager import CheckpointManager

        if slot.policy_uri:
            policy = CheckpointManager.load_from_uri(slot.policy_uri, policy_env_info, device)
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


class SlotControllerPolicy(Policy):
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
            self._device = torch.device(first_policy.device) if first_policy is not None else torch.device("cpu")
        self._agent_slot_map = agent_slot_map

        for idx, policy in slot_policies.items():
            if isinstance(policy, nn.Module):
                self.add_module(f"slot_{idx}", policy)

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return next(iter(self._slot_policies.values())).get_agent_experience_spec()

    def initialize_to_environment(self, policy_env_info, device: torch.device) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.initialize_to_environment(policy_env_info, device)
        self._device = torch.device(device)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        if "slot_id" not in td.keys():
            if self._agent_slot_map is None:
                raise RuntimeError("slot_id missing and no agent_slot_map provided for slot-aware policy routing")
            num_agents = self._agent_slot_map.numel()
            if num_agents == 0:
                raise RuntimeError("agent_slot_map cannot be empty for slot-aware policy routing")
            batch = td.batch_size.numel()
            if batch % num_agents != 0:
                raise RuntimeError(
                    f"slot-aware routing requires batch size ({batch}) to be divisible by num_agents ({num_agents})"
                )
            num_envs = batch // num_agents
            td.set("slot_id", self._agent_slot_map.to(device=td.device).repeat(num_envs))

        slot_ids = td.get("slot_id")
        assert isinstance(slot_ids, torch.Tensor), "slot_id must be a tensor"

        unique_ids: Iterable[int] = torch.unique(slot_ids).tolist()
        for b_id in unique_ids:
            mask = slot_ids == b_id

            sub_td = td[mask].clone()
            policy = self._slot_policies.get(int(b_id))
            assert policy is not None, f"No policy registered for slot id {int(b_id)}"

            out_td = policy.forward(sub_td, action=None if action is None else action[mask])

            for key in ("actions", "act_log_prob", "entropy", "values", "full_log_probs", "logits"):
                if key not in out_td.keys():
                    continue
                value = out_td.get(key)
                if key not in td.keys():
                    full_shape = td.batch_size + value.shape[1:]
                    td.set(key, value.new_zeros(full_shape))
                td.set_at_(key, value, mask)

        return td

    def reset_memory(self) -> None:  # noqa: D401
        for policy in self._slot_policies.values():
            policy.reset_memory()

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device
