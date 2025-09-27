"""Reusable head components for actor/critic/value projections."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from tensordict import TensorDict

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig


class LinearHeadConfig(ComponentConfig):
    """Configures a single linear projection with optional activation."""

    name: str = "linear_head"
    in_key: str
    out_key: str
    in_features: int
    out_features: int
    activation: Optional[str] = None
    manual_init: bool = False
    init_std: float = 1.0

    def make_component(self, env: Optional[object] = None) -> "LinearHead":
        return LinearHead(self)


class LinearHead(nn.Module):
    """Applies a linear layer to a TensorDict entry and writes the result to out_key."""

    def __init__(self, config: LinearHeadConfig) -> None:
        super().__init__()
        self.config = config

        linear = nn.Linear(config.in_features, config.out_features)
        if config.manual_init:
            nn.init.orthogonal_(linear.weight, config.init_std)
            nn.init.zeros_(linear.bias)
        else:
            linear = pufferlib.pytorch.layer_init(linear, std=config.init_std)

        self.linear = linear
        self.activation = self._build_activation(config.activation)

    def _build_activation(self, name: Optional[str]) -> Optional[nn.Module]:
        if not name:
            return None
        if not hasattr(nn, name):
            raise ValueError(f"Unknown activation '{name}' in torch.nn")
        module = getattr(nn, name)
        if not isinstance(module, type) or not issubclass(module, nn.Module):
            raise ValueError(f"Activation '{name}' is not a torch.nn.Module subclass")
        return module()

    def forward(self, td: TensorDict) -> TensorDict:
        input_tensor = td[self.config.in_key]
        output = self.linear(input_tensor)
        if self.activation is not None:
            output = self.activation(output)
        td.set(self.config.out_key, output)
        return td
