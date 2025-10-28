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
from metta.rl.training import PolicyEnvInterface
from mettagrid.config.mettagrid_config import ActionsConfig, Config
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
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
        return AgentClass(policy_env_info, self)


class Policy(TrainablePolicy, nn.Module):
    """Abstract base class defining the interface that all policies must implement.

    This class provides both the PyTorch nn.Module interface for training
    and the TrainablePolicy interface for compatibility with mettagrid Rollout.
    """

    def __init__(self, actions: ActionsConfig):
        TrainablePolicy.__init__(self, actions)
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

    def step(self, obs: AgentObservation) -> Action:
        """Get action from Policy."""
        # Convert observation to tensor dict format
        obs_array = np.array([obs])  # Add batch dimension
        td = self._obs_to_td(obs_array, self._policy.device)

        # Get action from policy
        self._policy(td)
        action = td["actions"][0].item()

        return action

    def reset(self) -> None:
        """Reset policy state if needed."""
        self._policy.reset_memory()

    def _obs_to_td(self, obs: np.ndarray, device: torch.device) -> TensorDict:
        """Convert observation array to TensorDict."""
        return TensorDict(
            {
                "env_obs": torch.from_numpy(obs).to(device),
                "dones": torch.zeros(len(obs), dtype=torch.float32, device=device),
                "truncateds": torch.zeros(len(obs), dtype=torch.float32, device=device),
            },
            batch_size=[len(obs)],
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
        game_rules: PolicyEnvInterface,
        actions: Optional[ActionsConfig] = None,
        box_obs: bool = True,
    ):
        if actions is None:
            actions = ActionsConfig()
        super().__init__(actions)
        self.policy = policy
        self._device = next(policy.parameters()).device if hasattr(policy, "parameters") else torch.device("cpu")
        if box_obs:
            self.obs_shaper = ObsShimBox(
                game_rules,
                config=ObsShimBoxConfig(in_key="env_obs", out_key="obs"),
            )
        else:
            self.obs_shaper = ObsShimTokens(
                game_rules,
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
