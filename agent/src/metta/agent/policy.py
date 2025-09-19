"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimTokens
from metta.rl.training.training_environment import EnvironmentMetaData
from mettagrid.config import Config
from mettagrid.util.module import load_symbol


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    components: List[ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: ComponentConfig

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
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
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


class ExternalPolicyWrapper(Policy):
    """
    For wrapping generic policies, aleiviating the need to conform to Metta's internal agent interface reqs.

    Expectations of the policy is that it takes a tensor of observations and returns a tensor of actions that matches
    the action space. That's to say that these policies will be used in evaluation, not in training.

    Policies that wish to be trained in metta should instead inherit from Policy and implement an agent experience spec,
    return the tensors needed for losses (ie values, entropy, and others depending on the loss), and the other methods
    if necessary.
    """

    def __init__(self, policy: nn.Module, env_metadata: EnvironmentMetaData, box_obs: bool = True):
        self.policy = policy
        if box_obs:
            self.obs_shaper = ObsShimBox(env=env_metadata, in_key="env_obs", out_key="obs")
        else:
            self.obs_shaper = ObsShimTokens(env=env_metadata, in_key="env_obs", out_key="obs")

    def forward(self, td: TensorDict) -> TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def get_agent_experience_spec(self) -> Composite:
        pass

    def initialize_to_environment(self, env_metadata: EnvironmentMetaData, device: torch.device):
        pass

    @property
    def device(self) -> torch.device:
        return self.policy.device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
