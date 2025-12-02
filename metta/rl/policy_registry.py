"""Minimal policy registry/loader for binding-based control."""

from __future__ import annotations

import json
from typing import Dict, Tuple

import torch

from metta.agent.policy import Policy
from metta.rl.binding_config import PolicyBindingConfig
from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol


class PolicyRegistry:
    """Loads and caches policies described by PolicyBindingConfig."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Policy] = {}

    def _cache_key(self, binding: PolicyBindingConfig) -> Tuple[str, str]:
        key_dict = {
            "uri": binding.policy_uri,
            "class_path": binding.class_path,
            "kwargs": binding.policy_kwargs,
        }
        return (binding.id, json.dumps(key_dict, sort_keys=True))

    def get(self, binding: PolicyBindingConfig, policy_env_info: PolicyEnvInterface, device: torch.device) -> Policy:
        """Instantiate (or reuse) a policy for the given binding."""

        if binding.use_trainer_policy:
            raise ValueError("use_trainer_policy bindings must be supplied externally, not loaded via registry")

        key = self._cache_key(binding)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if binding.policy_uri:
            policy = CheckpointManager.load_from_uri(binding.policy_uri, policy_env_info, device)
        elif binding.class_path:
            PolicyCls = load_symbol(binding.class_path)
            policy = PolicyCls(policy_env_info, **binding.policy_kwargs)
            if not isinstance(policy, Policy):
                raise TypeError(f"Loaded policy from {binding.class_path} is not a metta.agent.policy.Policy instance")
        else:
            raise ValueError("Binding must provide policy_uri or class_path to load a policy")

        policy = policy.to(device)
        policy.initialize_to_environment(policy_env_info, device)
        self._cache[key] = policy
        return policy
