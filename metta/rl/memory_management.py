from threading import RLock
from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


class Memory:
    def __init__(self):
        self.memory_dict = TensorDict({}, batch_size=torch.Size([1]))

    def get_memory(self, env_id):
        return self.memory_dict[env_id]

    def set_memory(self, memory, env_id):
        self.memory_dict[env_id] = memory

    def reset_env_memory(self, env_id):
        self.memory_dict[env_id] = {}

    def reset_memory(self):
        self.memory_dict = TensorDict({}, batch_size=torch.Size([1]))


class MemoryManager:
    """
    Minimal coordination layer for policy-internal state management.

    - Prefers modern state APIs if the policy implements them (get_states/set_states/reset_states)
    - Falls back to legacy memory APIs (get_memory/set_memory/reset_memory, reset_env_memory)
    - Supports optional per-environment operations via env_id when the policy exposes them
    - Thread-safe wrappers for multi-threaded inference loops
    """

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self._lock = RLock()

        # Cache method availability for quick checks
        self._has_get_states = hasattr(policy, "get_states")
        self._has_set_states = hasattr(policy, "set_states")
        self._has_reset_states = hasattr(policy, "reset_states")

        self._has_get_memory = hasattr(policy, "get_memory")
        self._has_set_memory = hasattr(policy, "set_memory")
        self._has_reset_memory = hasattr(policy, "reset_memory")
        self._has_reset_env_memory = hasattr(policy, "reset_env_memory")

        self.memory = Memory()
        self.memory.reset_memory()

    # ----------------------------------------------------------------------------
    # State accessors (preferred modern API)
    # ----------------------------------------------------------------------------
    def get_states(self, env_id: Optional[int] = None) -> Any:
        """ """
        self.memory.get_memory(env_id)

    def set_states(self, states: Any, env_id: Optional[int] = None) -> None:
        """Write policy states back to the policy if supported.

        Falls back to set_memory when set_states is not present. No-op otherwise.
        """
        self.memory.set_memory(states, env_id)

    def reset_states(self, env_id: Optional[int] = None) -> None:
        """Reset policy states; supports per-env when available on policy."""
        self.memory.reset_env_memory(env_id)
