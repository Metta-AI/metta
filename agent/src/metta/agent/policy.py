"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol
from metta.rl.training.training_environment import EnvironmentMetaData


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    def make_policy(self, env_metadata: EnvironmentMetaData) -> "Policy":
        """Create an agent instance from configuration."""

        AgentClass = load_symbol(self.class_path)
        return AgentClass(env_metadata, self)


class Policy(ABC, nn.Module):
    """Abstract base class defining the interface that all policies must implement.
    implement this interface."""

    @abstractmethod
    def forward(self, td: TensorDict) -> TensorDict:
        pass

    @property
    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(self, env_metadata: EnvironmentMetaData, device: torch.device):
        return

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def total_params(self) -> int: ...

    @abstractmethod
    def reset_memory(self):
        pass


# class PyTorchPolicyWrapper(Policy):
#     def __init__(self, policy: nn.Module):
#         self.policy = policy

#     def forward(self, td: TensorDict) -> TensorDict:
#         return self.policy(td)

#     def device(self) -> torch.device:
#         return self.policy.device
