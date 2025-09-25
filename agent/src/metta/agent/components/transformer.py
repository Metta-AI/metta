from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

from mettagrid.config import Config


class TransformerConfig(Config):
    latent_size: int = 128
    hidden_size: int = 32
    num_layers: int = 4
    nhead: int = 4
    ff_mult: int = 4
    max_seq_len: int = 16
    in_key: str = "encoded_obs"
    out_key: str = "hidden"
    kv_cache: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int) -> None:
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
        src = self.norm1(residual + attn_out)

        # Feed-forward block
        residual = src
        ff_out = self.ff(src)
        src = self.norm2(residual + ff_out)

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
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.use_kv_cache = self.config.kv_cache

        needs_proj = self.latent_size != self.hidden_size
        self.input_proj = nn.Linear(self.latent_size, self.hidden_size) if needs_proj else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pos_embedding = nn.Embedding(self.max_seq_len + 1, self.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.nhead, self.ff_mult) for _ in range(self.num_layers)]
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
            td[self.out_key] = h
        else:
            cls_token = self.cls_token.expand(-1, B, -1)  # [1, B, D]
            x = torch.cat([cls_token, x], dim=0)
            h = self._forward_training(x, x.shape[0], device)
            # Repeat cls token output for each timestep
            h = h.unsqueeze(0).expand(TT, -1, -1)
            td[self.out_key] = rearrange(h, "t b h -> (b t) h")

        return td

    def _forward_training(self, x: torch.Tensor, TT: int, device: torch.device) -> torch.Tensor:
        """Process a full sequence for training, using a causal mask."""
        pos_ids = torch.arange(TT, device=device)
        h = x + self.pos_embedding(pos_ids).unsqueeze(1)

        for layer in self.layers:
            h, _ = layer(h, past_kv=None)
        return h[0]

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

        # Vectorized cache reset
        reset_indices = env_ids[reset_flags].tolist()
        for env_id in reset_indices:
            self._kv_cache.pop(env_id, None)

        env_ids_list = env_ids.tolist()

        # 0. Initialize cache for new episodes with a CLS token pass
        new_episode_indices = [i for i, env_id in enumerate(env_ids_list) if env_id not in self._kv_cache]
        if new_episode_indices:
            B_new = len(new_episode_indices)
            new_env_ids = env_ids[new_episode_indices]
            cls_token = self.cls_token.expand(-1, B_new, -1)
            pos_ids = torch.tensor([0], device=device)
            h_cls = cls_token + self.pos_embedding(pos_ids).unsqueeze(1)

            for ll in range(self.num_layers):
                h_cls, new_kv = self.layers[ll](h_cls)
                k_batch, v_batch = new_kv
                ks_list = torch.chunk(k_batch, B_new, dim=0)
                vs_list = torch.chunk(v_batch, B_new, dim=0)
                for i, env_id in enumerate(new_env_ids.tolist()):
                    if env_id not in self._kv_cache:
                        self._kv_cache[env_id] = [None] * self.num_layers
                    self._kv_cache[env_id][ll] = (ks_list[i], vs_list[i])

        # 1. Vectorized cache gathering
        layer_past_kvs = []
        layer_padding_masks = []

        for ll in range(self.num_layers):
            batch_kvs = [self._kv_cache.get(env_id, [None] * self.num_layers)[ll] for env_id in env_ids_list]
            non_empty_kvs = [(i, kv) for i, kv in enumerate(batch_kvs) if kv is not None]

            if not non_empty_kvs:
                layer_past_kvs.append(None)
                layer_padding_masks.append(None)
                continue

            indices, kvs = zip(*non_empty_kvs, strict=False)
            indices = torch.tensor(indices, device=device)
            ks, vs = zip(*kvs, strict=False)

            ks_prep = [k.squeeze(0).permute(1, 0, 2) for k in ks]
            vs_prep = [v.squeeze(0).permute(1, 0, 2) for v in vs]

            padded_ks_prep = torch.nn.utils.rnn.pad_sequence(ks_prep, batch_first=False, padding_value=0.0)
            padded_vs_prep = torch.nn.utils.rnn.pad_sequence(vs_prep, batch_first=False, padding_value=0.0)

            S_max = padded_ks_prep.shape[0]
            padded_k = torch.zeros((B, self.nhead, S_max, self.d_head), device=device, dtype=x.dtype)
            padded_v = torch.zeros((B, self.nhead, S_max, self.d_head), device=device, dtype=x.dtype)

            padded_k.index_copy_(0, indices, padded_ks_prep.permute(1, 2, 0, 3))
            padded_v.index_copy_(0, indices, padded_vs_prep.permute(1, 2, 0, 3))

            lengths = torch.tensor([k.shape[2] for k in ks], device=device)
            padding_mask = torch.ones((B, S_max), dtype=torch.bool, device=device)
            mask_arange = torch.arange(S_max, device=device)
            scatter_mask = mask_arange.expand(len(lengths), S_max) < lengths.unsqueeze(1)
            padding_mask.scatter_(0, indices.unsqueeze(1).expand(-1, S_max), ~scatter_mask)

            layer_past_kvs.append((padded_k, padded_v))
            layer_padding_masks.append(padding_mask)

        # 2. Forward pass for the new observation
        h = x
        current_seq_pos = layer_past_kvs[0][0].shape[2] if layer_past_kvs[0] is not None else 0
        pos_ids = torch.arange(current_seq_pos, current_seq_pos + 1, device=device)
        h = h + self.pos_embedding(pos_ids).unsqueeze(1)

        new_kvs_by_layer = []
        for ll, layer in enumerate(self.layers):
            h, new_kv = layer(h, past_kv=layer_past_kvs[ll], key_padding_mask=layer_padding_masks[ll])
            new_kvs_by_layer.append(new_kv)

        # 3. Vectorized cache update
        for ll in range(self.num_layers):
            k_batch, v_batch = new_kvs_by_layer[ll]
            new_len = k_batch.shape[2]
            if new_len > self.max_seq_len:
                k_batch = k_batch[:, :, -self.max_seq_len :, :]
                v_batch = v_batch[:, :, -self.max_seq_len :, :]

            ks_list = torch.chunk(k_batch, B, dim=0)
            vs_list = torch.chunk(v_batch, B, dim=0)

            for i, env_id in enumerate(env_ids_list):
                self._kv_cache[env_id][ll] = (ks_list[i], vs_list[i])

        # 4. Get updated CLS token representation as final output
        cls_h = self.cls_token.expand(1, B, -1) + self.pos_embedding(torch.tensor([0], device=device)).unsqueeze(1)
        for ll, layer in enumerate(self.layers):
            # The new_kvs_by_layer already contains the full, padded cache including the latest observation
            past_kv = new_kvs_by_layer[ll]

            # The padding mask needs to be extended by one for the new (unpadded) observation token
            padding_mask = layer_padding_masks[ll]
            if padding_mask is not None:
                current_pad = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([padding_mask, current_pad], dim=1)

            cls_h, _ = layer(cls_h, past_kv=past_kv, key_padding_mask=padding_mask)

        return cls_h.squeeze(0)

    def get_memory(self):
        return self._kv_cache

    def set_memory(self, memory):
        self._kv_cache = memory if isinstance(memory, dict) else {}

    def reset_memory(self):
        self._kv_cache.clear()
