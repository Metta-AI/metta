from __future__ import annotations

import torch
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


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
        tokens = td[self.config.in_key]
        mask = td.get(self.config.mask_key, None)

        batch, seq_len, _ = tokens.shape
        max_tokens = self.config.max_tokens

        if max_tokens is not None:
            max_len = min(max_tokens, seq_len)
            trimmed_tokens = tokens[:, :max_len].contiguous()
            if mask is not None:
                new_mask = mask[:, :max_len].to(torch.bool)
            else:
                new_mask = torch.zeros((batch, max_len), dtype=torch.bool, device=tokens.device)
            trimmed_tokens = trimmed_tokens.masked_fill(new_mask.unsqueeze(-1), self.config.pad_value)
            lengths = None
        else:
            if mask is not None:
                valid_counts = (~mask.to(torch.bool)).sum(dim=1)
            else:
                valid_counts = torch.full((batch,), seq_len, device=tokens.device, dtype=torch.long)

            max_len = max(int(valid_counts.max().item()), 1)
            trimmed_tokens = tokens[:, :max_len].contiguous()

            positions = torch.arange(max_len, device=tokens.device).unsqueeze(0)
            capped_counts = valid_counts.clamp(max=max_len).unsqueeze(1)
            new_mask = positions >= capped_counts
            trimmed_tokens = trimmed_tokens.masked_fill(new_mask.unsqueeze(-1), self.config.pad_value)
            lengths = capped_counts.squeeze(1)

        td[self.config.out_key] = trimmed_tokens
        td[self.config.mask_key] = new_mask

        if self.config.length_key:
            if lengths is None:
                lengths = (~new_mask).sum(dim=1)
            td[self.config.length_key] = lengths

        return td
