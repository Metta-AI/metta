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
from mettagrid.policy.policy import MultiAgentPolicy, TrainablePolicy
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
        self._actions_by_id = self._policy_env_info.actions.actions()
        self._stateful_impl = self.make_stateful_policy_impl()

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

    def agent_step(self, agent_id: int, obs: AgentObservation) -> Action:
        td = self._obs_to_td(obs, self.device, agent_id=agent_id)
        result = self(td)
        if result is None:
            result = td
        action_source = result if "actions" in result.keys(True) else td
        action_index = int(action_source["actions"][0].item())
        return self._policy_env_info.actions.actions()[action_index]

    def agent_reset(self, agent_id: int, simulation: Optional[Simulation] = None) -> None:
        self.reset_memory()

    def _obs_to_td(self, obs: AgentObservation, device: torch.device, agent_id: int | None = None) -> TensorDict:
        obs_tensor = obs_to_obs_tensor(obs, self._policy_env_info.observation_space.shape, device)
        td = TensorDict(
            {
                "env_obs": obs_tensor,
                "dones": torch.zeros(1, dtype=torch.float32, device=device),
                "truncateds": torch.zeros(1, dtype=torch.float32, device=device),
                "bptt": torch.ones(1, dtype=torch.long, device=device),
            },
            batch_size=[1],
        )
        self._set_single_step_metadata(td, agent_slot=int(agent_id) if agent_id is not None else 0)
        return td

    def _set_single_step_metadata(self, td: TensorDict, *, agent_slot: int) -> None:
        ensure_sequence_metadata(td, batch_size=1, time_steps=1)
        device = td.device
        td.set("training_env_ids", torch.tensor([[agent_slot]], dtype=torch.long, device=device))
        td.set("row_id", torch.tensor([agent_slot], dtype=torch.long, device=device))
        td.set("t_in_row", torch.zeros(1, dtype=torch.long, device=device))


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


class _PolicyStateImpl(StatefulPolicyImpl[Any]):
    def __init__(self, policy: "Policy"):
        self._policy = policy
        self._active_agent_id: int | None = None

    def set_active_agent(self, agent_id: Optional[int]) -> None:
        self._active_agent_id = agent_id

    def reset(self) -> None:
        self._policy.reset_memory()

    def initial_agent_state(self) -> Any:
        return self._policy.initial_agent_state()

    def step_with_state(self, obs: AgentObservation, state: Any) -> tuple[Action, Any]:
        return self._policy.step_with_state(obs, state, agent_id=self._active_agent_id)
