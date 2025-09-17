from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.mettagrid.config import Config


class TransformerConfig(Config):
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    nhead: int = 4
    ff_mult: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    in_key: str = "encoded_obs"
    out_key: str = "hidden"


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,  # [T, B, D]
        attn_mask: Optional[torch.Tensor] = None,  # [T, T]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] True for PAD
    ) -> torch.Tensor:
        residual = src
        attn_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(residual + self.dropout(attn_out))

        residual = src
        ff_out = self.ff(src)
        src = self.norm2(residual + self.dropout(ff_out))
        return src


class Transformer(nn.Module):
    """
    Causal transformer layer that mirrors the LSTM layer's API, including shape handling
    and per-environment memory management. During rollout (TT=1) it maintains a per-env
    token cache (input embeddings) to provide autoregressive context. During training
    (TT>1) it processes full sequences with a causal mask and no cache.

    Notes
    - The __init__ of this layer (and MettaAgent) is only executed on fresh instantiation,
      not when reloading from a saved policy.
    - The cache stores projected input tokens (post input projection, pre-transformer) per env id.
    - We do not store KV caches per layer to keep implementation simple and robust to resets.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.nhead = self.config.nhead
        self.ff_mult = self.config.ff_mult
        self.max_seq_len = self.config.max_seq_len
        self.dropout_p = self.config.dropout
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key

        needs_proj = self.latent_size != self.hidden_size
        self.input_proj = nn.Linear(self.latent_size, self.hidden_size) if needs_proj else nn.Identity()

        self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(self.hidden_size, self.nhead, self.ff_mult, self.dropout_p)
                for _ in range(self.num_layers)
            ]
        )

        # Per-environment token caches (projected inputs). Keyed by absolute env id (int)
        self._token_cache: Dict[int, torch.Tensor] = {}

    def __setstate__(self, state):
        """Ensure caches are re-initialized after loading from checkpoint."""
        self.__dict__.update(state)
        if not hasattr(self, "_token_cache"):
            self._token_cache = {}
        self._token_cache.clear()

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # True/inf above diagonal -> mask future positions
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    @torch._dynamo.disable  # Exclude forward from Dynamo to avoid graph breaks from Python dict usage
    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_key]  # [BT, latent]

        TT = 1
        B = td.batch_size.numel()
        if td.get("bptt", None) is not None and td["bptt"][0] != 1:
            TT = int(td["bptt"][0].item())
        B = B // TT

        x = rearrange(x, "(b t) h -> t b h", b=B, t=TT)  # [T, B, latent]
        x = self.input_proj(x)  # [T, B, hidden]

        # Environment ids for double-buffered envs
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None:
            env_ids = training_env_ids.reshape(-1)  # [B]
        else:
            training_env_id = td.get("training_env_id", None)
            if training_env_id is not None:
                start = int(training_env_id.reshape(-1)[0].item())
                env_ids = torch.arange(start, start + B, device=x.device)
            else:
                env_ids = torch.arange(B, device=x.device)

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        reset_flags: Optional[torch.Tensor]
        if dones is not None and truncateds is not None:
            if TT == 1:
                reset_flags = (dones.bool() | truncateds.bool()).reshape(B)
            else:
                # For training we assume segments are episode-consistent; reset not applied here
                reset_flags = None
        else:
            reset_flags = None

        device = x.device

        if TT == 1:
            # Rollout: maintain per-env caches and run with padding + causal mask
            # Prepare per-b sequences of variable lengths by concatenating cache and current token
            seqs = []
            lengths = []
            for b in range(B):
                env_id = int(env_ids[b].item())
                if reset_flags is not None and bool(reset_flags[b].item()):
                    past = torch.empty((0, self.hidden_size), device=device, dtype=x.dtype)
                else:
                    past = self._token_cache.get(
                        env_id, torch.empty((0, self.hidden_size), device=device, dtype=x.dtype)
                    )

                cur = x[0, b]  # [hidden]
                seq = torch.cat([past, cur.unsqueeze(0)], dim=0)  # [L_i, hidden]
                if seq.size(0) > self.max_seq_len:
                    seq = seq[-self.max_seq_len :]
                seqs.append(seq)
                lengths.append(seq.size(0))

            Lmax = max(lengths) if lengths else 1
            # Build padded batch [Lmax, B, hidden]
            src = x.new_zeros((Lmax, B, self.hidden_size))
            key_padding_mask = torch.ones((B, Lmax), dtype=torch.bool, device=device)  # True = PAD
            for b in range(B):
                Lb = lengths[b]
                if Lb > 0:
                    src[:Lb, b] = seqs[b]
                    key_padding_mask[b, :Lb] = False

            # Add positional embeddings
            pos_ids = torch.arange(Lmax, device=device)
            src = src + self.pos_embedding(pos_ids).unsqueeze(1)

            attn_mask = self._causal_mask(Lmax, device)
            h = src
            for layer in self.layers:
                h = layer(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

            # Gather last valid token output per batch element
            out = x.new_empty((B, self.hidden_size))
            for b in range(B):
                out[b] = h[lengths[b] - 1, b]

            # Update caches with new input token appended (detached)
            for b in range(B):
                env_id = int(env_ids[b].item())
                seq = seqs[b].detach()
                self._token_cache[env_id] = seq

            td[self.out_key] = out  # [B, hidden]
            td[self.out_key] = rearrange(td[self.out_key], "b h -> (b) h")  # [BT, hidden]
            return td

        # Training: process full sequence [TT, B, hidden] with causal mask; no caches used
        attn_mask = self._causal_mask(TT, device)
        pos_ids = torch.arange(TT, device=device)
        h = x + self.pos_embedding(pos_ids).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask, key_padding_mask=None)

        td[self.out_key] = rearrange(h, "t b h -> (b t) h")  # [BT, hidden]
        return td

    def get_memory(self):
        return self._token_cache

    def set_memory(self, memory):
        """Cannot be called at the MettaAgent level - use policy.component[this_layer_name].set_memory()"""
        self._token_cache = memory if isinstance(memory, dict) else {}

    def reset_memory(self):
        self._token_cache.clear()
