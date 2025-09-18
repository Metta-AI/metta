"""Policy base classes and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimTokens
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from metta.rl.training.training_environment import EnvironmentMetaData


class PolicyArchitecture(Config):
    """Configuration container for constructing policies."""

    class_path: str
    components: List[ComponentConfig] = []
    action_probs_config: ComponentConfig

    def make_policy(self, env_metadata: "EnvironmentMetaData"):
        AgentClass = load_symbol(self.class_path)
        return AgentClass(env_metadata, self)


class Policy(ABC, nn.Module):
    """Abstract base class defining the policy interface."""

    @abstractmethod
    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        """Forward pass used by losses / rollout."""
        raise NotImplementedError

    def get_agent_experience_spec(self) -> Composite:
        """Default experience spec; concrete policies may override."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(self, env_metadata, device: torch.device) -> None:  # noqa: D401
        """Hook for env-specific initialization (optional)."""
        return None

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def total_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset_memory(self) -> None:
        raise NotImplementedError


class ExternalPolicyWrapper(Policy):
    """
    For wrapping generic policies, aleiviating the need to conform to Metta's internal agent interface reqs.

    Expectations of the policy is that it takes a tensor of observations and returns a tensor of actions that matches
    the action space. That's to say that these policies will be used in evaluation, not in training.

    Policies that wish to be trained in metta should instead inherit from Policy and implement an agent experience spec,
    return the tensors needed for losses (ie values, entropy, and others depending on the loss), and the other methods
    if necessary.
    """

    def __init__(self, policy: torch.nn.Module, env_metadata: "EnvironmentMetaData", box_obs: bool = True):
        self.policy = policy
        if box_obs:
            self.obs_shaper = ObsShimBox(env=env_metadata, in_key="env_obs", out_key="obs")
        else:
            self.obs_shaper = ObsShimTokens(env=env_metadata, in_key="env_obs", out_key="obs")

    def forward(self, td: TensorDict) -> TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def get_agent_experience_spec(self):
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
