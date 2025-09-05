from typing import List, Optional

import pufferlib.pytorch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential

from metta.common.config.config import Config


class MLPConfig(Config):
    """Configuration for a multi-layer perceptron (MLP)."""

    name: str
    in_key: str
    out_key: str
    in_features: int
    out_features: int
    hidden_features: List[int]
    nonlinearity: Optional[str] = "ReLU"  # e.g., "ReLU", "Tanh"; Name of a torch.nn module
    output_nonlinearity: Optional[str] = None  # e.g., "ReLU", "Tanh"; Name of a torch.nn module
    layer_init_std: float = 1.0


class MLP(nn.Module):
    """A flexible MLP module using TensorDict."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        layers = []
        current_in_features = self.config.in_features
        current_in_key = self.config.in_key

        all_dims = self.config.hidden_features + [self.config.out_features]

        for i, out_features in enumerate(all_dims):
            is_last_layer = i == len(all_dims) - 1

            linear_layer = pufferlib.pytorch.layer_init(
                nn.Linear(current_in_features, out_features), std=self.config.layer_init_std
            )

            if is_last_layer:
                layer_out_key = self.config.out_key
            else:
                # Nested key under the MLP's name
                layer_out_key = (self.config.name, f"layer_{i}_out")

            tdm = TDM(linear_layer, in_keys=[current_in_key], out_keys=[layer_out_key])
            layers.append(tdm)

            if is_last_layer:
                if self.config.output_nonlinearity:
                    nonlinearity_module = self._get_nonlinearity(self.config.output_nonlinearity)
                    layers.append(TDM(nonlinearity_module, in_keys=[layer_out_key], out_keys=[layer_out_key]))
            elif self.config.nonlinearity:
                nonlinearity_module = self._get_nonlinearity(self.config.nonlinearity)
                # Apply nonlinearity in-place on the output key of the linear layer
                layers.append(TDM(nonlinearity_module, in_keys=[layer_out_key], out_keys=[layer_out_key]))

            current_in_features = out_features
            current_in_key = layer_out_key

        self.network = TensorDictSequential(*layers)

    def _get_nonlinearity(self, name: str) -> nn.Module:
        if hasattr(nn, name):
            cls = getattr(nn, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls()
        raise ValueError(f"Unsupported or unknown nonlinearity in torch.nn: {name}")

    def forward(self, td: TensorDict) -> TensorDict:
        return self.network(td)
