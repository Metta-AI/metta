"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from __future__ import annotations

import abc
import typing

import pydantic
import tensordict
import torch
import torch.nn as nn
import torch.nn.parallel
import torchrl.data

import metta.agent.components.component_config as component_config
import metta.agent.components.obs_shim as obs_shim
import metta.rl.utils as rl_utils
import mettagrid.base_config
import mettagrid.policy.policy as policy_module
import mettagrid.policy.policy_env_interface as policy_env_interface
import mettagrid.simulator
import mettagrid.util.module


class PolicyArchitecture(mettagrid.base_config.Config):
    """Policy architecture configuration."""

    class_path: str

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True)

    components: list[component_config.ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: component_config.ComponentConfig

    def make_policy(self, policy_env_info: policy_env_interface.PolicyEnvInterface) -> "Policy":
        """Create an agent instance from configuration."""

        agent_cls = mettagrid.util.module.load_symbol(self.class_path)
        return agent_cls(policy_env_info, self)  # type: ignore[misc]


class Policy(policy_module.TrainablePolicy, nn.Module):
    """Abstract base class defining the interface that all policies must implement."""

    def __init__(self, policy_env_info: policy_env_interface.PolicyEnvInterface):
        policy_module.TrainablePolicy.__init__(self, policy_env_info)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(
        self, td: tensordict.TensorDict, action: typing.Optional[torch.Tensor] = None
    ) -> tensordict.TensorDict:
        raise NotImplementedError

    def get_agent_experience_spec(self) -> torchrl.data.Composite:
        """Return the policy's required experience spec."""

        return torchrl.data.Composite(
            env_obs=torchrl.data.UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(
        self, policy_env_info: policy_env_interface.PolicyEnvInterface, device: torch.device
    ) -> None:
        return

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @property
    def total_params(self) -> int:
        """Count trainable parameters for logging/metrics."""

        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @abc.abstractmethod
    def reset_memory(self) -> None:
        pass

    def network(self) -> nn.Module:
        """Return the nn.Module representing the policy."""
        return self

    def agent_policy(self, agent_id: int) -> policy_module.AgentPolicy:
        """Return an AgentPolicy adapter for the specified agent index."""
        return _SingleAgentAdapter(self, agent_id)


class _SingleAgentAdapter(policy_module.AgentPolicy):
    """Adapter to provide AgentPolicy interface for a single agent from a multi-agent Policy."""

    def __init__(self, policy: Policy, agent_id: int):
        super().__init__(policy._policy_env_info)
        self._policy = policy
        self._agent_id = agent_id
        self._actions_by_id = self._policy_env_info.actions.actions()

    def step(self, obs: mettagrid.simulator.AgentObservation) -> mettagrid.simulator.Action:
        """Get action from Policy."""
        td = self._obs_to_td(obs, self._policy.device)
        self._policy(td)
        return self._actions_by_id[int(td["actions"][0].item())]

    def reset(self, simulation: typing.Optional[mettagrid.simulator.Simulation] = None) -> None:
        """Reset policy state if needed."""
        self._policy.reset_memory()

    def _obs_to_td(
        self, obs: mettagrid.simulator.AgentObservation, device: torch.device
    ) -> tensordict.TensorDict:
        tokens = [token.value for token in obs.tokens]
        obs_tensor = torch.tensor(tokens, dtype=torch.uint8).unsqueeze(0).to(device)

        td = tensordict.TensorDict(
            {
                "env_obs": obs_tensor,
                "dones": torch.zeros(1, dtype=torch.float32, device=device),
                "truncateds": torch.zeros(1, dtype=torch.float32, device=device),
                "bptt": torch.ones(1, dtype=torch.long, device=device),
            },
            batch_size=[1],
        )

        rl_utils.ensure_sequence_metadata(td, batch_size=1, time_steps=1)
        return td


class DistributedPolicy(torch.nn.parallel.DistributedDataParallel, Policy):
    """Thin wrapper around DistributedDataParallel that preserves Policy interface."""

    module: Policy

    def __init__(self, policy: Policy, device: torch.device):
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
    """Wrapper for generic policies that don't conform to Metta's internal agent interface."""

    def __init__(
        self,
        policy: nn.Module,
        policy_env_interface: policy_env_interface.PolicyEnvInterface,
        box_obs: bool = True,
    ) -> None:
        super().__init__(policy_env_interface)
        self.policy = policy
        self._device = next(policy.parameters()).device if hasattr(policy, "parameters") else torch.device("cpu")
        shim_config = obs_shim.ObsShimBoxConfig if box_obs else obs_shim.ObsShimTokensConfig
        shim_cls = obs_shim.ObsShimBox if box_obs else obs_shim.ObsShimTokens
        self.obs_shaper = shim_cls(
            policy_env_interface,
            config=shim_config(in_key="env_obs", out_key="obs"),
        )

    def forward(self, td: tensordict.TensorDict) -> tensordict.TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def initialize_to_environment(
        self, game_rules: policy_env_interface.PolicyEnvInterface, device: torch.device
    ) -> None:
        return

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self) -> None:
        return
