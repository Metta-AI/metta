"""Minimal policy registry/loader for slot-based control."""

from __future__ import annotations

import json
from typing import Dict, Tuple

import torch

from metta.agent.policy import Policy
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.slot_config import PolicySlotConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol


class SlotRegistry:
    """Loads and caches policies described by PolicySlotConfig."""

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
        """Instantiate (or reuse) a policy for the given slot."""

        if slot.use_trainer_policy:
            raise ValueError("use_trainer_policy slots must be supplied externally, not loaded via registry")

        key = self._cache_key(slot, device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

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
