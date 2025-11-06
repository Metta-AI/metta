"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

import abc
import typing

import pydantic
import tensordict
import torch
import torch.nn as nn
import torch.nn.parallel
import torchrl.data

import metta.agent.components.component_config
import metta.agent.components.obs_shim
import metta.rl.utils
import mettagrid.base_config
import mettagrid.policy.policy
import mettagrid.policy.policy_env_interface
import mettagrid.simulator
import mettagrid.util.module


class PolicyArchitecture(mettagrid.base_config.Config):
    """Policy architecture configuration."""

    class_path: str

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True)

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: metta.agent.components.component_config.ComponentConfig

    def make_policy(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface) -> "Policy":
        """Create an agent instance from configuration."""

        AgentClass = mettagrid.util.module.load_symbol(self.class_path)
        return AgentClass(policy_env_info, self)  # type: ignore[misc]


class Policy(mettagrid.policy.policy.TrainablePolicy, nn.Module):
    """Abstract base class defining the interface that all policies must implement.

    This class provides both the PyTorch nn.Module interface for training
    and the TrainablePolicy interface for compatibility with mettagrid Rollout.
    """

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface):
        mettagrid.policy.policy.TrainablePolicy.__init__(self, policy_env_info)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(self, td: tensordict.TensorDict, action: typing.Optional[torch.Tensor] = None) -> tensordict.TensorDict:
        pass

    def get_agent_experience_spec(self) -> torchrl.data.Composite:
        """Return the policy's required experience spec."""

        return torchrl.data.Composite(
            env_obs=torchrl.data.UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(
        self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface, device: torch.device
    ):
        return

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @property
    def total_params(self) -> int:
        """Count trainable parameters for logging/metrics."""

        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @abc.abstractmethod
    def reset_memory(self):
        pass

    def network(self) -> nn.Module:
        """Return the nn.Module representing the policy."""
        return self

    def agent_policy(self, agent_id: int) -> mettagrid.policy.policy.AgentPolicy:
        """Return an AgentPolicy adapter for the specified agent index."""
        return _SingleAgentAdapter(self, agent_id)


class _SingleAgentAdapter(mettagrid.policy.policy.AgentPolicy):
    """Adapter to provide AgentPolicy interface for a single agent from a multi-agent Policy."""

    def __init__(self, policy: "Policy", agent_id: int):
        super().__init__(policy._policy_env_info)
        self._policy = policy
        self._agent_id = agent_id
        self._actions_by_id = self._policy_env_info.actions.actions()

    def step(self, obs: mettagrid.simulator.AgentObservation) -> mettagrid.simulator.Action:
        """Get action from Policy."""
        # Convert observation to tensor dict format
        td = self._obs_to_td(obs, self._policy.device)

        # Get action from policy
        self._policy(td)
        return self._actions_by_id[int(td["actions"][0].item())]

    def reset(self) -> None:
        """Reset policy state if needed."""
        self._policy.reset_memory()

    def _obs_to_td(self, obs: mettagrid.simulator.AgentObservation, device: torch.device) -> tensordict.TensorDict:
        """Convert AgentObservation to TensorDict."""
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

        metta.rl.utils.ensure_sequence_metadata(td, batch_size=1, time_steps=1)

        return td


class DistributedPolicy(torch.nn.parallel.DistributedDataParallel):
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
        policy_env_interface: mettagrid.policy.policy_env_interface.PolicyEnvInterface,
        box_obs: bool = True,
    ):
        super().__init__(policy_env_interface)
        self.policy = policy
        self._device = next(policy.parameters()).device if hasattr(policy, "parameters") else torch.device("cpu")
        if box_obs:
            self.obs_shaper = metta.agent.components.obs_shim.ObsShimBox(
                policy_env_interface,
                config=metta.agent.components.obs_shim.ObsShimBoxConfig(in_key="env_obs", out_key="obs"),
            )
        else:
            self.obs_shaper = metta.agent.components.obs_shim.ObsShimTokens(
                policy_env_interface,
                config=metta.agent.components.obs_shim.ObsShimTokensConfig(in_key="env_obs", out_key="obs"),
            )

    def forward(self, td: tensordict.TensorDict) -> tensordict.TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

    def initialize_to_environment(
        self, game_rules: mettagrid.policy.policy_env_interface.PolicyEnvInterface, device: torch.device
    ):
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
