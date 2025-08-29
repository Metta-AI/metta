from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


class Memory:
    """In-process memory store keyed by env_id using a TensorDict backend."""

    def __init__(self):
        self.memory_dict = TensorDict({}, batch_size=torch.Size([1]))

    def get_memory(self, env_id):
        """Return memory for a specific env_id (or key)."""
        return self.memory_dict[env_id]["states"]

    def set_memory(self, memory, env_id):
        """Set memory for a specific env_id (or key)."""
        self.memory_dict[env_id]["states"] = memory

    def reset_env_memory(self, env_id):
        """Clear memory for a specific env_id (or key)."""
        self.memory_dict[env_id] = {}

    def reset_memory(self):
        """Clear all stored memory entries."""
        self.memory_dict = TensorDict({}, batch_size=torch.Size([1]))


class MemoryManager:
    """Minimal, thread-safe coordinator for policy memory/state payloads."""

    def __init__(self, policy: nn.Module):
        self.policy = policy

        self.memory = Memory()
        self.memory.reset_memory()

    def get_states(self, env_id: Optional[int] = None) -> Any:
        """Get stored memory for env_id (or global if None)."""
        self.memory.get_memory(env_id)

    def set_states(self, states: Any, env_id: Optional[int] = None) -> None:
        """Persist memory for env_id (or global if None)."""

        # print(f"set_states: {states}")
        self.memory.set_memory(states, env_id)

    def reset_states(self, env_id: Optional[int] = None) -> None:
        """Reset stored memory for env_id (or all if None)."""
        print(f"reset_states: {env_id}")
        self.memory.reset_env_memory(env_id)
