from __future__ import annotations

import torch
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class ObsTokenCompactConfig(ComponentConfig):
    """Compacts observation tokens so that valid entries appear first and truncates to a fixed budget."""

    in_key: str
    out_key: str
    max_tokens: int = 64
    pad_value: float = 0.0
    name: str = "obs_token_compact"

    def make_component(self, env=None):
        return ObsTokenCompact(config=self)


class ObsTokenCompact(torch.nn.Module):
    """Reorders tokens so valid entries (mask == False) are contiguous and trims to ``max_tokens``."""

    def __init__(self, config: ObsTokenCompactConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, td: TensorDict) -> TensorDict:
        tokens = td[self.config.in_key]
        mask = td.get("obs_mask")

        batch_size, seq_len, feat_dim = tokens.shape
        target_k = min(self.config.max_tokens, seq_len)

        if mask is None:
            compact_tokens = tokens[:, :target_k]
            compact_mask = torch.zeros(
                batch_size,
                target_k,
                dtype=torch.bool,
                device=tokens.device,
            )
        else:
            mask_bool = mask.to(torch.bool)
            seq_indices = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            # Push masked positions to the end while preserving order
            sort_keys = seq_indices + mask_bool.long() * seq_len
            sorted_idx = torch.argsort(sort_keys, dim=1)

            gather_idx = sorted_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
            sorted_tokens = torch.gather(tokens, 1, gather_idx)
            sorted_mask = torch.gather(mask_bool, 1, sorted_idx)

            compact_tokens = sorted_tokens[:, :target_k]
            compact_mask = sorted_mask[:, :target_k]

            valid_counts = (~mask_bool).sum(dim=1).clamp(max=target_k)
            if target_k > 0:
                pad_positions = (
                    torch.arange(target_k, device=tokens.device).unsqueeze(0)
                    >= valid_counts.unsqueeze(1)
                )
                compact_mask = compact_mask | pad_positions
                if self.config.pad_value != 0.0:
                    compact_tokens = compact_tokens.masked_fill(
                        pad_positions.unsqueeze(-1),
                        self.config.pad_value,
                    )
                else:
                    compact_tokens = compact_tokens.masked_fill(
                        pad_positions.unsqueeze(-1),
                        0.0,
                    )

        td[self.config.out_key] = compact_tokens
        td["obs_mask"] = compact_mask
        return td
