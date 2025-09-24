from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class ObsTokenPoolConfig(ComponentConfig):
    in_key: str
    out_key: str
    feat_dim: int
    hidden_dim: int
    pool: Literal["mean", "max"] = "mean"
    activation: bool = True
    name: str = "obs_token_pool"

    def make_component(self, env=None):
        return ObsTokenPool(config=self)


class ObsTokenPool(nn.Module):
    def __init__(self, config: ObsTokenPoolConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(self.config.feat_dim, self.config.hidden_dim)
        self.activation = nn.GELU() if self.config.activation else nn.Identity()

    def forward(self, td: TensorDict) -> TensorDict:
        tokens = td[self.config.in_key]
        mask = td.get("obs_mask")

        hidden = self.activation(self.linear(tokens))

        if mask is None:
            if self.config.pool == "max":
                pooled = hidden.max(dim=1).values
            else:
                pooled = hidden.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1)
            valid = ~mask_expanded

            if self.config.pool == "max":
                masked_hidden = hidden.masked_fill(~valid, float("-inf"))
                pooled = masked_hidden.max(dim=1).values
                pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0)
            else:
                masked_hidden = hidden.masked_fill(~valid, 0.0)
                counts = valid.sum(dim=1).clamp(min=1)
                pooled = masked_hidden.sum(dim=1) / counts

        td[self.config.out_key] = pooled
        return td
