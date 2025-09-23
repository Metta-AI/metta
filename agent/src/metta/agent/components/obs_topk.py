from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class ObsTokenTopKConfig(ComponentConfig):
    in_key: str
    out_key: str
    k: int = 128
    keep_pad: bool = False
    name: str = "obs_token_topk"

    def make_component(self, env=None):
        return ObsTokenTopK(config=self)


@dataclass
class TopKOutput:
    tokens: torch.Tensor
    mask: torch.Tensor


class ObsTokenTopK(nn.Module):
    def __init__(self, config: ObsTokenTopKConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, td: TensorDict) -> TensorDict:
        tokens = td[self.config.in_key]  # (B, M, 3)
        mask = td.get("obs_mask")  # (B, M)

        values = tokens[..., 2].to(torch.float32).abs()
        mask_bool = None
        if mask is not None:
            mask_bool = mask.to(torch.bool)
            values = values.masked_fill(mask_bool, float("-inf"))

        k = min(self.config.k, tokens.size(1))
        topk_vals, indices = torch.topk(values, k=k, dim=1)

        gathered_tokens = torch.gather(
            tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)),
        )

        if k < self.config.k:
            pad_size = self.config.k - k
            pad_tokens = torch.zeros(
                tokens.size(0),
                pad_size,
                tokens.size(-1),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            gathered_tokens = torch.cat([gathered_tokens, pad_tokens], dim=1)

        valid = torch.isfinite(topk_vals)
        gathered_mask = ~valid
        if self.config.keep_pad and mask_bool is not None:
            gathered_mask = torch.gather(mask_bool, dim=1, index=indices) | gathered_mask

        if gathered_tokens.size(1) > valid.size(1):
            pad = gathered_tokens.size(1) - valid.size(1)
            gathered_mask = torch.cat(
                [gathered_mask, torch.ones(tokens.size(0), pad, device=tokens.device, dtype=torch.bool)],
                dim=1,
            )

        td[self.config.out_key] = gathered_tokens
        td["obs_mask"] = gathered_mask
        return td
