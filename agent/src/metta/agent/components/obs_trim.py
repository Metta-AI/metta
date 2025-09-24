from __future__ import annotations

import torch
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.util.profile import PROFILER


class ObsTokenTrimConfig(ComponentConfig):
    """Trim trailing padded tokens based on ``obs_mask`` while preserving order."""

    in_key: str
    out_key: str
    max_tokens: int | None = None
    mask_key: str = "obs_mask"
    length_key: str | None = "obs_token_lengths"
    pad_value: float = 0.0
    name: str = "obs_token_trim"

    def make_component(self, env=None):
        return ObsTokenTrim(config=self)


class ObsTokenTrim(torch.nn.Module):
    def __init__(self, config: ObsTokenTrimConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, td: TensorDict) -> TensorDict:
        with PROFILER.section("obs_token_trim"):
            tokens = td[self.config.in_key]
            mask = td.get(self.config.mask_key, None)

            batch, seq_len, feat_dim = tokens.shape

            if mask is not None:
                mask_bool = mask.to(torch.bool)
                valid_counts = (~mask_bool).sum(dim=1)
            else:
                mask_bool = None
                valid_counts = torch.full((batch,), seq_len, device=tokens.device, dtype=torch.long)

            max_len = int(valid_counts.max().item()) if valid_counts.numel() else seq_len
            if self.config.max_tokens is not None:
                max_len = min(max_len, self.config.max_tokens)

            if max_len == 0:
                max_len = 1

            trimmed_tokens = tokens[:, :max_len].contiguous()

            expanded_indices = torch.arange(max_len, device=tokens.device).unsqueeze(0)
            capped_counts = valid_counts.clamp(max=max_len).unsqueeze(1)
            new_mask = expanded_indices >= capped_counts

            trimmed_tokens = trimmed_tokens.masked_fill(new_mask.unsqueeze(-1), self.config.pad_value)

        td[self.config.out_key] = trimmed_tokens
        td[self.config.mask_key] = new_mask

        if self.config.length_key:
            td[self.config.length_key] = capped_counts.squeeze(1)

        return td
