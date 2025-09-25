from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

from mettagrid.config import Config


class TransformerConfig(Config):
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    nhead: int = 4
    ff_mult: int = 4
    max_seq_len: int = 16
    dropout: float = 0.0
    in_key: str = "encoded_obs"
    out_key: str = "hidden"
    kv_cache: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_head = d_model // nhead
        assert self.d_head * nhead == d_model, "d_model must be divisible by nhead"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

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
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S_past]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        T, B, D = src.shape
        residual = src

        # Self-attention block with Post-LN, matching original structure
        q, k, v = self.qkv_proj(src).chunk(3, dim=-1)

        q = q.view(T, B, self.nhead, self.d_head).permute(1, 2, 0, 3)  # B, h, T, d_h
        k = k.view(T, B, self.nhead, self.d_head).permute(1, 2, 0, 3)  # B, h, T, d_h
        v = v.view(T, B, self.nhead, self.d_head).permute(1, 2, 0, 3)  # B, h, T, d_h

        is_training = past_kv is None
        if not is_training:  # Rollout with KV cache
            past_k, past_v = past_kv  # B, h, S_past, d_h
            k = torch.cat([past_k, k], dim=2)  # B, h, S_past+T, d_h
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k.detach(), v.detach())

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask is [B, S_past]. Pad for current token(s) (always unpadded)
            current_pad = torch.zeros((B, T), dtype=torch.bool, device=key_padding_mask.device)
            full_key_padding_mask = torch.cat([key_padding_mask, current_pad], dim=1)  # [B, S_past+T]
            # S->D attention mask requires (N, S) shape. For batched multi-head, (N, num_heads, S) is not supported
            # but we can use a broadcastable mask (N, 1, 1, S)
            attn_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_past+T]

        # Use is_causal for training, and attn_mask for rollout padding
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_training and attn_mask is None
        )

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(T, B, D)
        attn_out = self.out_proj(attn_output)
        src = self.norm1(residual + self.dropout(attn_out))

        # Feed-forward block
        residual = src
        ff_out = self.ff(src)
        src = self.norm2(residual + self.dropout(ff_out))

        return src, new_kv


class Transformer(nn.Module):
    """
    Causal transformer layer that mirrors the LSTM layer's API.
    During rollout (TT=1), it maintains a per-environment K/V cache for each layer
    to provide efficient autoregressive context. During training (TT>1), it processes
    full sequences with a causal mask and no cache.

    The cache is managed internally but can be accessed via get/set_memory,
    allowing the training loop to persist cache states across rollout/training boundaries.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.nhead = self.config.nhead
        self.ff_mult = self.config.ff_mult
        self.max_seq_len = self.config.max_seq_len
        self.dropout_p = self.config.dropout
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.use_kv_cache = self.config.kv_cache

        needs_proj = self.latent_size != self.hidden_size
        self.input_proj = nn.Linear(self.latent_size, self.hidden_size) if needs_proj else nn.Identity()

        self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(self.hidden_size, self.nhead, self.ff_mult, self.dropout_p)
                for _ in range(self.num_layers)
            ]
        )

        # Per-environment K/V caches, keyed by absolute env id.
        # List is over layers: [(k0, v0), (k1, v1), ...]
        self._kv_cache: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def __setstate__(self, state):
        """Ensure caches are re-initialized after loading from checkpoint."""
        self.__dict__.update(state)
        if not hasattr(self, "_kv_cache"):
            self._kv_cache = {}
        self._kv_cache.clear()

    @torch._dynamo.disable  # Exclude forward from Dynamo to avoid graph breaks from Python dict usage
    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_key]

        TT = 1
        B = td.batch_size[0] if td.batch_size else 1
        if "bptt" in td.keys() and td.get("bptt", None) is not None and td["bptt"][0] != 1:
            TT = int(td["bptt"][0].item())
        B = B // TT

        # Reshape to [T, B, ...]
        if x.ndim == 2:  # [BT, latent]
            x = rearrange(x, "(b t) h -> t b h", b=B, t=TT)
        elif x.ndim == 3:  # [BT, S, latent] -> [T, B, S, latent]
            x = rearrange(x, "(b t) s h -> t b s h", b=B, t=TT)
            # Combine S into T for sequence processing
            x = rearrange(x, "t b s h -> (t s) b h")
            TT = x.shape[0]
        x = self.input_proj(x)

        # Handle env_ids for cache indexing
        if "training_env_ids" in td.keys() and td.get("training_env_ids", None) is not None:
            env_ids = td["training_env_ids"].reshape(-1)
        else:
            env_ids = torch.arange(B, device=x.device)

        device = x.device
        is_rollout = TT == 1 and self.use_kv_cache

        if is_rollout:
            h = self._forward_rollout(x, B, env_ids, td.get("dones"), td.get("truncateds"))
            td[self.out_key] = rearrange(h, "b h -> (b) h")
        else:
            h = self._forward_training(x, TT, device)
            td[self.out_key] = rearrange(h, "t b h -> (b t) h")

        return td

    def _forward_training(self, x: torch.Tensor, TT: int, device: torch.device) -> torch.Tensor:
        """Process a full sequence for training, using a causal mask."""
        pos_ids = torch.arange(TT, device=device)
        h = x + self.pos_embedding(pos_ids).unsqueeze(1)

        for layer in self.layers:
            h, _ = layer(h, past_kv=None)
        return h

    def _forward_rollout(
        self,
        x: torch.Tensor,
        B: int,
        env_ids: torch.Tensor,
        dones: Optional[torch.Tensor],
        truncateds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Process a single step for rollout, using the K/V cache."""
        device = x.device
        # Check for resets to clear cache
        reset_flags = (
            (dones.bool() | truncateds.bool()).reshape(B)
            if dones is not None and truncateds is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )

        # 1. Prepare layer-wise batched past_kv and padding masks
        layer_past_kvs = []
        layer_padding_masks = []
        max_len = 0

        for ll in range(self.num_layers):
            ks, vs = [], []
            lengths = []
            for b in range(B):
                env_id = env_ids[b].item()
                if reset_flags[b].item():
                    self._kv_cache.pop(env_id, None)

                past_kv = self._kv_cache.get(env_id, [None] * self.num_layers)[ll]
                if past_kv is not None:
                    k, v = past_kv  # k: [1, h, L, d]
                    ks.append(k.squeeze(0))  # h, L, d
                    vs.append(v.squeeze(0))
                    lengths.append(k.shape[2])
                else:
                    lengths.append(0)

            max_len = max(lengths) if lengths else 0
            if max_len == 0:
                layer_past_kvs.append(None)
                layer_padding_masks.append(None)
                continue

            # Pad and batch
            padded_k = torch.zeros((B, self.nhead, max_len, self.d_head), device=device, dtype=x.dtype)
            padded_v = torch.zeros((B, self.nhead, max_len, self.d_head), device=device, dtype=x.dtype)
            padding_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)

            k_idx, v_idx = 0, 0
            for b in range(B):
                if lengths[b] > 0:
                    padded_k[b, :, : lengths[b], :] = ks[k_idx]
                    padded_v[b, :, : lengths[b], :] = vs[v_idx]
                    padding_mask[b, : lengths[b]] = False
                    k_idx += 1
                    v_idx += 1
            layer_past_kvs.append((padded_k, padded_v))
            layer_padding_masks.append(padding_mask)

        # 2. Forward pass through layers
        h = x
        seq_len = max_len + 1
        pos_ids = torch.arange(seq_len - 1, seq_len, device=device)
        h = h + self.pos_embedding(pos_ids).unsqueeze(1)

        new_kvs_by_layer = []
        for ll, layer in enumerate(self.layers):
            h, new_kv = layer(h, past_kv=layer_past_kvs[ll], key_padding_mask=layer_padding_masks[ll])
            new_kvs_by_layer.append(new_kv)

        # 3. Update cache
        for b in range(B):
            env_id = env_ids[b].item()
            env_kvs = []
            for ll in range(self.num_layers):
                k, v = new_kvs_by_layer[ll]  # [B, h, L_new, d]
                # Unpad and truncate
                len_past = lengths[b] if max_len > 0 else 0
                new_len = len_past + 1
                unpadded_k = k[b : b + 1, :, :new_len, :]
                unpadded_v = v[b : b + 1, :, :new_len, :]
                if new_len > self.max_seq_len:
                    unpadded_k = unpadded_k[:, :, -self.max_seq_len :, :]
                    unpadded_v = unpadded_v[:, :, -self.max_seq_len :, :]
                env_kvs.append((unpadded_k, unpadded_v))
            self._kv_cache[env_id] = env_kvs

        return h.squeeze(0)  # [1, B, D] -> [B, D]

    def get_memory(self):
        return self._kv_cache

    def set_memory(self, memory):
        self._kv_cache = memory if isinstance(memory, dict) else {}

    def reset_memory(self):
        self._kv_cache.clear()
