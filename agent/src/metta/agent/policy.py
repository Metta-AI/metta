"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import abstractmethod
from typing import ClassVar, List, Optional

import numpy as np
import torch
import torch.nn as nn
from pydantic import ConfigDict
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.obs_shim import (
    ObsShimBox,
    ObsShimBoxConfig,
    ObsShimTokens,
    ObsShimTokensConfig,
)
from mettagrid.config.mettagrid_config import Config
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation
from mettagrid.util.module import load_symbol


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    components: List[ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: ComponentConfig

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> "Policy":
        """Create an agent instance from configuration."""

        AgentClass = load_symbol(self.class_path)
        return AgentClass(policy_env_info, self)  # type: ignore[misc]


class Policy(TrainablePolicy, nn.Module):
    """Abstract base class defining the interface that all policies must implement.

    This class provides both the PyTorch nn.Module interface for training
    and the TrainablePolicy interface for compatibility with mettagrid Rollout.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface):
        TrainablePolicy.__init__(self, policy_env_info)
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        pass

    def get_agent_experience_spec(self) -> Composite:
        """Return the policy's required experience spec."""

        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device):
        return

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    def total_params(self) -> int:
        """Count trainable parameters for logging/metrics."""

        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @abstractmethod
    def reset_memory(self):
        pass

    def network(self) -> nn.Module:
        """Get the underlying neural network for training.

        Since Policy is itself an nn.Module, return self.
        """
        return self

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            An AgentPolicy instance for this agent
        """
        return _SingleAgentAdapter(self, agent_id)


class _SingleAgentAdapter(AgentPolicy):
    """Adapter to provide AgentPolicy interface for a single agent from a multi-agent Policy."""

    def __init__(self, policy: "Policy", agent_id: int):
        super().__init__(policy._actions)
        self._policy = policy
        self._agent_id = agent_id
        self._actions_by_id = self._actions.actions()

    def step(self, obs: AgentObservation) -> Action:
        """Get action from Policy."""
        # Convert observation to tensor dict format
        td = self._obs_to_td(obs, self._policy.device)

        # Get action from policy
        self._policy(td)
        return self._actions_by_id[int(td["actions"][0].item())]

    def reset(self) -> None:
        """Reset policy state if needed."""
        self._policy.reset_memory()

    def _obs_to_td(self, obs: AgentObservation, device: torch.device) -> TensorDict:
        """Convert AgentObservation to TensorDict."""
        tokens = []

        for token in obs.tokens:
            col, row = token.location
            # Pack coordinates into a single byte: first 4 bits are col, last 4 bits are row
            coords_byte = ((col & 0x0F) << 4) | (row & 0x0F)
            feature_id = token.feature.id
            value = token.value
            tokens.append([coords_byte, feature_id, value])

        # Pad to max_tokens with [0xFF, 0, 0] (end-of-tokens marker)
        while len(tokens) < 200:
            tokens.append([0xFF, 0, 0])

        # Convert to numpy array and then to tensor: [M, 3] -> [1, M, 3]
        obs_array = np.array(tokens, dtype=np.uint8)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)

        return TensorDict(
            {
                "env_obs": obs_tensor,
                "dones": torch.zeros(1, dtype=torch.float32, device=device),
                "truncateds": torch.zeros(1, dtype=torch.float32, device=device),
            },
            batch_size=[1],
        )


class DistributedPolicy(DistributedDataParallel):
    """Thin wrapper around DistributedDataParallel that preserves Policy interface."""

    module: "Policy"

    def __init__(self, policy: "Policy", device: torch.device):
        kwargs = {
            "module": policy,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cpu" or device.index is None:
            super().__init__(**kwargs)
        else:
            kwargs.update({"device_ids": [device.index], "output_device": device.index})
            super().__init__(**kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ExternalPolicyWrapper(Policy):
    """Wrapper for generic policies that don't conform to Metta's internal agent interface.

    Expectations of the policy is that it takes a tensor of observations and returns a tensor of actions that matches
    the action space. These policies will be used in evaluation, not in training.

    Policies that wish to be trained in metta should instead inherit from Policy and implement an agent experience spec,
    return the tensors needed for losses (ie values, entropy, and others depending on the loss), and the other methods
    if necessary.
    """

    def __init__(
        self,
        policy: nn.Module,
        policy_env_interface: PolicyEnvInterface,
        box_obs: bool = True,
    ):
        super().__init__(policy_env_interface)
        self.policy = policy
        self._device = next(policy.parameters()).device if hasattr(policy, "parameters") else torch.device("cpu")
        if box_obs:
            self.obs_shaper = ObsShimBox(
                policy_env_interface,
                config=ObsShimBoxConfig(in_key="env_obs", out_key="obs"),
            )
        else:
            self.obs_shaper = ObsShimTokens(
                policy_env_interface,
                config=ObsShimTokensConfig(in_key="env_obs", out_key="obs"),
            )

    def forward(self, td: TensorDict) -> TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def initialize_to_environment(self, game_rules: PolicyEnvInterface, device: torch.device):
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
