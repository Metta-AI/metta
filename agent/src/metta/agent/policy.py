"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.obs_shim import (
    ObsShimBox,
    ObsShimBoxConfig,
    ObsShimTokens,
    ObsShimTokensConfig,
)
from metta.agent.policy_architecture import PolicyArchitecture
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.policy.lstm import obs_to_obs_tensor
from mettagrid.policy.policy import (
    AgentPolicy,
    MultiAgentPolicy,
    StatefulAgentPolicy,
    StatefulPolicyImpl,
)
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import PolicyRegistryABCMeta
from mettagrid.simulator import Action, AgentObservation


class Policy(MultiAgentPolicy, nn.Module):
    """Abstract base class defining the interface that all policies must implement.

    This class provides both the PyTorch nn.Module interface for training
    and the MultiAgentPolicy interface for compatibility with mettagrid Rollout.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface):
        MultiAgentPolicy.__init__(self, policy_env_info)
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

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load network weights from file using PyTorch state dict."""
        import torch

        self.load_state_dict(torch.load(policy_data_path, map_location=self.device))

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save network weights to file using torch.save."""
        import torch

        torch.save(self.state_dict(), policy_data_path)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return an AgentPolicy adapter for the specified agent index."""
        return StatefulAgentPolicy(self._stateful_impl, self._policy_env_info, agent_id)

    def make_stateful_policy_impl(self) -> StatefulPolicyImpl[Any]:
        return _PolicyStateImpl(self)

    def initial_agent_state(self) -> Any:
        return None

    def load_agent_state(self, state: Any | None) -> None:
        if state is not None:
            raise RuntimeError(f"{self.__class__.__name__} does not store per-agent state")

    def dump_agent_state(self) -> Any:
        return None

    def step_with_state(self, obs: AgentObservation, state: Any, agent_id: int | None = None) -> tuple[Action, Any]:
        td = self._obs_to_td(obs, self.device, agent_id)
        if state is not None:
            self.load_agent_state(state)
        self(td)
        new_state = self.dump_agent_state()
        action_idx = int(td["actions"][0].item())
        return self._actions_by_id[action_idx], new_state

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


class CheckpointPolicy(Policy):
    """Policy wrapper for checkpoint bundles with architecture specs and safetensors weights."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        architecture_spec: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(policy_env_info)
        self._architecture_spec = architecture_spec
        self._device = torch.device(device)
        architecture = PolicyArchitecture.from_spec(architecture_spec)
        self._policy = architecture.make_policy(policy_env_info).to(self._device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "_policy":
                raise
            policy = self.__dict__.get("_policy")
            if policy is None and "_modules" in self.__dict__:
                policy = self.__dict__["_modules"].get("_policy")
            if policy is None:
                raise
            return getattr(policy, name)

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        return self._policy.forward(td, action=action)

    @property
    def device(self) -> torch.device:
        return self._policy.device

    def reset_memory(self) -> None:
        self._policy.reset_memory()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def load_policy_data(self, policy_data_path: str) -> None:
        state_dict = load_safetensors_file(str(Path(policy_data_path).expanduser()))
        self._policy.load_state_dict(dict(state_dict))
        initialize = getattr(self._policy, "initialize_to_environment", None)
        if callable(initialize):
            initialize(self._policy_env_info, self._device)

    def state_dict(self, *args, **kwargs):
        return self._policy.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self._policy.load_state_dict(state_dict, *args, **kwargs)

    def save_policy_data(self, policy_data_path: str) -> None:
        self._policy.save_policy_data(policy_data_path)

    def network(self) -> Optional[nn.Module]:
        return self._policy.network()

    def reset(self) -> None:
        self._policy.reset()

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._policy.step_batch(raw_observations, raw_actions)


class DistributedPolicy(MultiAgentPolicy, DistributedDataParallel, metaclass=PolicyRegistryABCMeta):
    """Thin wrapper around DistributedDataParallel that preserves Policy interface."""

    def __init__(self, policy: MultiAgentPolicy, device: torch.device):
        MultiAgentPolicy.__init__(self, policy.policy_env_info)

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

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)


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
