from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential
from torchrl.modules import ConsistentDropout

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig


class MLPConfig(ComponentConfig):
    """Variable depth MLP. You don't have to set input feats since it uses lazy linear.
    Don't set an output nonlinearity if it's used as an output head!"""

    in_key: str
    out_key: str
    out_features: int
    name: str = "mlp"
    hidden_features: List[int]
    in_features: Optional[int] = None
    nonlinearity: Optional[str] = "ReLU"  # e.g., "ReLU", "Tanh"; Name of a torch.nn module
    output_nonlinearity: Optional[str] = None  # e.g., "ReLU", "Tanh"; Name of a torch.nn module
    layer_init_std: float = 1.0
    dropout_p: float = 0.25
    is_dropout: bool = True

    def make_component(self, env=None):
        return MLP(config=self)


class MLP(nn.Module):
    """A flexible MLP module using TensorDict."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.mask = None

        layers = []
        current_in_features = self.config.in_features
        if current_in_features is None:
            raise ValueError(
                "MLPConfig.in_features must be set so layers can be initialized before distributed wrapping."
            )

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

        if self.config.is_dropout:
            self.dropout = ConsistentDropout(p=self.config.dropout_p)

    def _get_nonlinearity(self, name: str) -> nn.Module:
        if hasattr(nn, name):
            cls = getattr(nn, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls()
        raise ValueError(f"Unsupported or unknown nonlinearity in torch.nn: {name}")

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.network(td)
        if self.config.is_dropout:
            # Apply dropout to the output key
            output_tensor = td[self.config.out_key]
            # Check if mask needs to be reset due to batch size mismatch
            if self.mask is not None and self.mask.shape[0] != output_tensor.shape[0]:
                self.mask = None
            result = self.dropout(output_tensor, mask=self.mask)
            # ConsistentDropout returns (tensor, mask) in train mode, but only tensor in eval mode
            if isinstance(result, tuple):
                dropped_tensor, mask = result
                self.mask = mask
            else:
                dropped_tensor = result
            td[self.config.out_key] = dropped_tensor
        return td


###------------- Deep Residual MLP -------------------------
class DeepResMLPConfig(ComponentConfig):
    in_key: str
    out_key: str
    depth: int
    in_features: int
    name: str = "deep_res_mlp"

    def make_component(self, env=None):
        return ResNetMLP(config=self)


class ResidualBlock(nn.Module):
    """
    This class is instantiated by ResNetMLP.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Swish(nn.Module):
    """
    This is a helper class for ResNetMLP.
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class ResNetMLP(nn.Module):
    """
    Applies a residual dense connection to the incoming data: y = x + F(x)
    Input and output shapes are the same. To scale, you need to create an initial and/or final linear layer separately
    that maps to the hidden size you want.
    """

    def __init__(self, config: DeepResMLPConfig):
        super().__init__()
        self.config = config
        hidden_size = self.config.in_features
        self._depth = self.config.depth

        if self._depth % 4 != 0:
            raise ValueError("Depth must be a multiple of 4.")
        self._num_blocks = self._depth // 4

        self.residual_layers = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(self._num_blocks)])

    def forward(self, td: TensorDict):
        x = td[self.config.in_key]
        x = self.residual_layers(x)
        td[self.config.out_key] = x
        return td


# ------------- End Deep Residual MLP ----------------------------------
