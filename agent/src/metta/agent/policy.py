"""PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on."""

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional

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
from metta.rl.training import GameRules
from mettagrid.base_config import Config
from mettagrid.util.module import load_symbol

PolicyPresetFactory = Callable[[], type["PolicyArchitecture"]]


def _resolve_symbol(path: str) -> Any:
    try:
        return load_symbol(path)
    except (AttributeError, ModuleNotFoundError, ValueError) as exc:
        raise ValueError(f"Failed to resolve symbol '{path}': {exc}") from exc


def _preset(module: str, attribute: str) -> PolicyPresetFactory:
    return lambda: getattr(importlib.import_module(module), attribute)


def _normalize_preset_key(value: str) -> str:
    return value.lower().replace("-", "_")


POLICY_PRESETS: Dict[str, PolicyPresetFactory] = {
    "fast": _preset("metta.agent.policies.fast", "FastConfig"),
    "fast_dynamics": _preset("metta.agent.policies.fast_dynamics", "FastDynamicsConfig"),
    "fast_lstm_reset": _preset("metta.agent.policies.fast_lstm_reset", "FastLSTMResetConfig"),
    "memory_free": _preset("metta.agent.policies.memory_free", "MemoryFreeConfig"),
    "puffer": _preset("metta.agent.policies.puffer", "PufferPolicyConfig"),
    "transformer": _preset("metta.agent.policies.transformer", "TransformerPolicyConfig"),
    "vit": _preset("metta.agent.policies.vit", "ViTDefaultConfig"),
    "vit_reset": _preset("metta.agent.policies.vit_reset", "ViTResetConfig"),
    "vit_sliding_trans": _preset("metta.agent.policies.vit_sliding_trans", "ViTSlidingTransConfig"),
}


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    components: List[ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: ComponentConfig

    @classmethod
    def resolve(cls, value: Any) -> "PolicyArchitecture":
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            preset_key = _normalize_preset_key(value)
            preset = POLICY_PRESETS.get(preset_key)
            if preset is None:
                available = ", ".join(sorted(POLICY_PRESETS))
                raise ValueError(f"Unknown policy preset: {value}. Available: [{available}]")
            return preset()()

        if isinstance(value, type) and issubclass(value, cls):
            return value()

        raise TypeError(f"Unable to resolve value {value!r} into a {cls.__name__}")

    def make_policy(self, game_rules: GameRules) -> "Policy":
        """Create an agent instance from configuration."""

        agent_cls = _resolve_symbol(self.class_path)
        return agent_cls(env_metadata, self)

class Policy(ABC, nn.Module):
    """Abstract base class defining the interface that all policies must implement.
    implement this interface."""

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

    def initialize_to_environment(self, game_rules: GameRules, device: torch.device):
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

    def __init__(self, policy: nn.Module, game_rules: GameRules, box_obs: bool = True):
        super().__init__()
        self.policy = policy
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

    def initialize_to_environment(self, game_rules: GameRules, device: torch.device):
        pass

    @property
    def device(self) -> torch.device:
        return self.policy.device

    @property
    def total_params(self) -> int:
        return 0

    def reset_memory(self):
        pass
