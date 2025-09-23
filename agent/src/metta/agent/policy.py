"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import torch
import torch.nn as nn
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
from metta.rl.training import EnvironmentMetaData
from mettagrid.config import Config
from mettagrid.util.module import load_symbol


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    components: List[ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: ComponentConfig

    _ALIASES: ClassVar[Dict[str, str]] = {}
    _CANONICAL_ALIASES: ClassVar[Dict[str, str]] = {}

    @classmethod
    def register_alias(cls, alias: str, target: str) -> None:
        cls._ALIASES[alias] = target
        cls._ALIASES[alias.casefold()] = target
        cls._CANONICAL_ALIASES[alias] = target

    @classmethod
    def available_aliases(cls) -> dict[str, str]:
        return dict(cls._CANONICAL_ALIASES)

    @classmethod
    def resolve(cls, value: Any) -> "PolicyArchitecture":
        if isinstance(value, cls):
            return value

        if isinstance(value, type) and issubclass(value, cls):
            return value()

        if isinstance(value, str):
            reference = cls._ALIASES.get(value)
            if reference is None:
                reference = cls._ALIASES.get(value.casefold())
            if reference is None:
                reference = value

            try:
                symbol = load_symbol(reference)
            except (ImportError, AttributeError, ModuleNotFoundError) as exc:
                if reference is value:
                    aliases = cls._format_aliases()
                    raise ValueError(f"Unknown policy preset '{value}'. Valid options: {aliases}") from exc
                raise ValueError(f"Unable to load policy preset '{value}': {exc}") from exc

            return cls._coerce_resolved(symbol, reference)

        if callable(value):
            candidate = value()
            if isinstance(candidate, cls):
                return candidate

        raise TypeError(f"Unable to resolve value {value!r} into an instance of {cls.__name__}")

    @classmethod
    def _coerce_resolved(cls, resolved: Any, reference: str) -> "PolicyArchitecture":
        if isinstance(resolved, cls):
            return resolved

        if isinstance(resolved, type) and issubclass(resolved, cls):
            return resolved()

        if callable(resolved):
            candidate = resolved()
            if isinstance(candidate, cls):
                return candidate

        raise TypeError(f"Resolved object {resolved!r} for reference {reference!r} is not a {cls.__name__}")

    @classmethod
    def _format_aliases(cls) -> str:
        aliases = sorted(cls._CANONICAL_ALIASES)
        return ", ".join(aliases) if aliases else "(none available)"

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

    def get_agent_experience_spec(self) -> Composite:
        """Return the policy's required experience spec."""

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
    def total_params(self) -> int:
        """Count trainable parameters for logging/metrics."""

        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @abstractmethod
    def reset_memory(self):
        pass


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
    """
    For wrapping generic policies, aleiviating the need to conform to Metta's internal agent interface reqs.

    Expectations of the policy is that it takes a tensor of observations and returns a tensor of actions that matches
    the action space. That's to say that these policies will be used in evaluation, not in training.

    Policies that wish to be trained in metta should instead inherit from Policy and implement an agent experience spec,
    return the tensors needed for losses (ie values, entropy, and others depending on the loss), and the other methods
    if necessary.
    """

    def __init__(self, policy: nn.Module, env_metadata: EnvironmentMetaData, box_obs: bool = True):
        super().__init__()
        self.policy = policy
        if box_obs:
            self.obs_shaper = ObsShimBox(
                env_metadata,
                config=ObsShimBoxConfig(in_key="env_obs", out_key="obs"),
            )
        else:
            self.obs_shaper = ObsShimTokens(
                env_metadata,
                config=ObsShimTokensConfig(in_key="env_obs", out_key="obs"),
            )

    def forward(self, td: TensorDict) -> TensorDict:
        self.obs_shaper(td)
        return self.policy(td["obs"])

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


_POLICY_ALIAS_TARGETS: Dict[str, str] = {
    "fast": "metta.agent.policies.fast.FastConfig",
    "fast_lstm_reset": "metta.agent.policies.fast_lstm_reset.FastLSTMResetConfig",
    "fast_dynamics": "metta.agent.policies.fast_dynamics.FastDynamicsConfig",
    "cnn_trans": "metta.agent.policies.cnn_trans.CNNTransConfig",
    "vit_small": "metta.agent.policies.vit.ViTSmallConfig",
    # Backwards compatibility alias for existing docs/CLI usage.
    "vit": "metta.agent.policies.vit.ViTSmallConfig",
}


for alias, target in _POLICY_ALIAS_TARGETS.items():
    PolicyArchitecture.register_alias(alias, target)
