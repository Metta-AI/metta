"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import abstractmethod
from typing import ClassVar, List, Optional

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
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.base_config import Config
from mettagrid.policy.lstm import obs_to_obs_tensor
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import PolicyRegistryABCMeta
from mettagrid.simulator import Action, AgentObservation, Simulation
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
        """Return the nn.Module representing the policy."""
        return self

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return an AgentPolicy adapter for the specified agent index."""
        return _SingleAgentAdapter(self, agent_id)


class _SingleAgentAdapter(AgentPolicy):
    """Adapter to provide AgentPolicy interface for a single agent from a multi-agent Policy."""

    def __init__(self, policy: "Policy", agent_id: int):
        super().__init__(policy._policy_env_info)
        self._policy = policy
        self._agent_id = agent_id
        self._actions_by_id = self._policy_env_info.actions.actions()

    def step(self, obs: AgentObservation) -> Action:
        """Get action from Policy."""
        # Convert observation to tensor dict format
        td = self._obs_to_td(obs, self._policy.device)

        # Get action from policy
        self._policy(td)
        return self._actions_by_id[int(td["actions"][0].item())]

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        """Reset policy state if needed."""
        self._policy.reset_memory()

    def _obs_to_td(self, obs: AgentObservation, device: torch.device) -> TensorDict:
        """Convert AgentObservation to TensorDict."""

        obs_tensor = obs_to_obs_tensor(obs, self._policy_env_info.observation_space.shape, device)

        td = TensorDict(
            {
                "env_obs": obs_tensor,
                "dones": torch.zeros(1, dtype=torch.float32, device=device),
                "truncateds": torch.zeros(1, dtype=torch.float32, device=device),
                "bptt": torch.ones(1, dtype=torch.long, device=device),
                "training_env_ids": torch.tensor([[self._agent_id]], dtype=torch.long, device=device),
            },
            batch_size=[1],
        )

        ensure_sequence_metadata(td, batch_size=1, time_steps=1)
        return td


class DistributedPolicy(TrainablePolicy, DistributedDataParallel, metaclass=PolicyRegistryABCMeta):
    """Thin wrapper around DistributedDataParallel that preserves Policy interface."""

    def __init__(self, policy: MultiAgentPolicy, device: torch.device):
        TrainablePolicy.__init__(self, policy.policy_env_info)

        # Then initialize DistributedDataParallel
        kwargs = {
            "module": policy,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type != "cpu" and device.index is not None:
            kwargs.update({"device_ids": [device.index], "output_device": device.index})
        DistributedDataParallel.__init__(self, **kwargs)

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

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device):
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
