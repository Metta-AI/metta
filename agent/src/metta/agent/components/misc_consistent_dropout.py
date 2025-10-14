from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous
from torchrl.modules import ConsistentDropout

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig


class MLPConsistentDropoutConfig(ComponentConfig):
    """Variable depth MLP with consistent dropout for RL policy gradients.

    Consistent dropout ensures the same dropout mask is used during both
    rollout and gradient computation, preventing bias in policy gradients.
    """

    in_key: str
    out_key: str
    out_features: int
    name: str = "mlp_consistent_dropout"
    hidden_features: List[int]
    in_features: Optional[int] = None
    nonlinearity: Optional[str] = "ReLU"
    output_nonlinearity: Optional[str] = None
    layer_init_std: float = 1.0
    dropout_p: float = 0.2

    def make_component(self, env=None):
        return MLPConsistentDropout(config=self)


class MLPConsistentDropout(nn.Module):
    """A flexible MLP module with consistent dropout using TensorDict.

    Consistent dropout maintains the same mask across rollout and training,
    preventing biased policy gradients in RL settings.
    """

    def __init__(self, config: MLPConsistentDropoutConfig):
        super().__init__()
        self.config = config

        current_in_features = self.config.in_features
        if current_in_features is None:
            raise ValueError(
                "MLPConsistentDropoutConfig.in_features must be set so layers can be "
                "initialized before distributed wrapping."
            )

        all_dims = self.config.hidden_features + [self.config.out_features]

        self.linears = nn.ModuleList()
        self.nonlinearities = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i, out_features in enumerate(all_dims):
            is_last_layer = i == len(all_dims) - 1

            linear_layer = pufferlib.pytorch.layer_init(
                nn.Linear(current_in_features, out_features), std=self.config.layer_init_std
            )
            self.linears.append(linear_layer)

            if is_last_layer:
                if self.config.output_nonlinearity:
                    self.nonlinearities.append(self._get_nonlinearity(self.config.output_nonlinearity))
                else:
                    self.nonlinearities.append(None)
                self.dropouts.append(None)
            else:
                if self.config.nonlinearity:
                    self.nonlinearities.append(self._get_nonlinearity(self.config.nonlinearity))
                else:
                    self.nonlinearities.append(None)

                if self.config.dropout_p > 0:
                    self.dropouts.append(ConsistentDropout(p=self.config.dropout_p))
                else:
                    self.dropouts.append(None)

            current_in_features = out_features

    def _get_nonlinearity(self, name: str) -> nn.Module:
        if hasattr(nn, name):
            cls = getattr(nn, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls()
        raise ValueError(f"Unsupported or unknown nonlinearity in torch.nn: {name}")

    def get_agent_experience_spec(self) -> Composite:
        """Return experience spec for dropout masks that need to be stored in replay buffer."""
        spec = Composite()

        if self.config.dropout_p > 0:
            for i in range(len(self.config.hidden_features)):
                mask_key = (self.config.name, f"dropout_mask_{i}")
                hidden_dim = self.config.hidden_features[i]

                spec[mask_key] = UnboundedContinuous(
                    shape=torch.Size([hidden_dim]),
                    dtype=torch.float32,
                )

        return spec

    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.config.in_key]

        for i, (linear, nonlinearity, dropout) in enumerate(
            zip(self.linears, self.nonlinearities, self.dropouts, strict=False)
        ):
            x = linear(x)

            if nonlinearity is not None:
                x = nonlinearity(x)

            if dropout is not None:
                mask_key = (self.config.name, f"dropout_mask_{i}")
                mask = td.get(mask_key, None)

                if self.training:
                    x, mask = dropout(x, mask=mask)
                    td[mask_key] = mask
                else:
                    x = dropout(x, mask=mask)

        td[self.config.out_key] = x
        return td
