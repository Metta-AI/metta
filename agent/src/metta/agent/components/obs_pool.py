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
        compiler_mod = getattr(torch, "compiler", None)
        mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if mark_step is not None:
            mark_step()

        tokens = td[self.config.in_key]
        mask = td.get("obs_mask")

        if self.config.pool == "max":
            hidden = self.activation(self.linear(tokens))
            if mask is None:
                pooled = hidden.max(dim=1).values
            else:
                mask_expanded = mask.unsqueeze(-1)
                valid = ~mask_expanded
                masked_hidden = hidden.masked_fill(~valid, float("-inf"))
                pooled = masked_hidden.max(dim=1).values
                pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0)
        else:  # mean pooling path
            if mask is None:
                pooled_tokens = tokens.mean(dim=1)
            else:
                mask_expanded = mask.unsqueeze(-1)
                valid = ~mask_expanded
                masked_tokens = tokens.masked_fill(~valid, 0.0)
                counts = valid.sum(dim=1).clamp(min=1)
                pooled_tokens = masked_tokens.sum(dim=1) / counts
            hidden = self.linear(pooled_tokens)
            pooled = self.activation(hidden)

        td[self.config.out_key] = pooled
        return td
