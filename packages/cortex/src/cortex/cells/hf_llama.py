"""LLaMA decoder-layer wrapper that calls the HF layer and manages Cache/positions."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.core import update_parent_state
from cortex.cells.registry import register_cell
from cortex.config import CellConfig
from cortex.types import MaybeState, ResetMask, Tensor
from transformers.cache_utils import DynamicCache


class HFLlamaLayerConfig(CellConfig):
    """Config for the HF LLaMA layer wrapper; expects hf_layer, hf_submodel, hf_config set at runtime."""

    # Tag used by the registry
    cell_type: str = "hf_llama_layer"

    # Interface
    hidden_size: int | None = None
    mem_len: int = 0


@register_cell(HFLlamaLayerConfig)
class HFLlamaLayerCell(MemoryCell):
    """Wrap a single HF LLaMA decoder layer as a Cortex MemoryCell."""

    def __init__(self, cfg: HFLlamaLayerConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("HFLlamaLayerConfig.hidden_size must be set")
        super().__init__(hidden_size=int(cfg.hidden_size))
        self.cfg = cfg

        # Extra runtime objects (accepted via pydantic extra fields)
        hf_layer = getattr(cfg, "hf_layer", None)
        hf_submodel = getattr(cfg, "hf_submodel", None)
        hf_config = getattr(cfg, "hf_config", None)
        if not isinstance(hf_layer, nn.Module):
            raise ValueError("HFLlamaLayerConfig must include an 'hf_layer' nn.Module")
        if not isinstance(hf_submodel, nn.Module):
            raise ValueError("HFLlamaLayerConfig must include an 'hf_submodel' nn.Module (e.g., model.model)")
        self.hf_layer: nn.Module = hf_layer
        self.hf_submodel: nn.Module = hf_submodel
        self.hf_config = hf_config

        self.mem_len = int(cfg.mem_len)

        self._CacheCls = DynamicCache

    # ---- state helpers ----
    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        cache = self._CacheCls(config=self.hf_config) if self.hf_config is not None else self._CacheCls()
        pos = torch.zeros(batch, dtype=torch.long, device=device)
        seg_pos = torch.zeros(batch, dtype=torch.long, device=device)
        return TensorDict({"cache": cache, "pos": pos, "seg_pos": seg_pos}, batch_size=[batch])

    def _normalize_resets(self, resets: Optional[ResetMask], B: int, T: int, device: torch.device) -> torch.Tensor:
        if resets is None:
            return torch.zeros(B, T, device=device, dtype=torch.long)
        r = resets.to(device=device)
        if r.dtype == torch.bool:
            r = r.long()
        elif r.dtype.is_floating_point:
            r = (r > 0).long()
        else:
            r = r.long()
        if r.shape == (B,) and T == 1:
            r = r.view(B, 1)
        if r.shape != (B, T):
            raise ValueError(f"resets must have shape {(B, T)}; got {r.shape}")
        return r

    @staticmethod
    def _unwrap_cache(cache_obj: object) -> object:
        """Return underlying HF cache if wrapped by TensorDict NonTensorData."""
        if cache_obj is not None and hasattr(cache_obj, "data") and "NonTensor" in cache_obj.__class__.__name__:
            return getattr(cache_obj, "data", cache_obj)
        return cache_obj

    def _layer_past_len(self, cache: DynamicCache, layer_idx: int) -> int:
        """Return past length for this layer from the HF cache."""
        if hasattr(cache, "layers") and layer_idx < len(cache.layers):
            layer = cache.layers[layer_idx]
            if getattr(layer, "is_initialized", False):
                return int(layer.get_seq_length())
        return 0

    @staticmethod
    def _build_last_reset_indices(resets_bt: torch.Tensor) -> torch.Tensor:
        """Vectorized last reset index per timestep; -1 if none so far."""
        B, T = resets_bt.shape
        device = resets_bt.device
        ar = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        tagged = torch.where(resets_bt != 0, ar, torch.full_like(ar, -1))
        out = torch.cummax(tagged, dim=1).values
        return out

    def _build_additive_mask(
        self,
        *,
        B: int,
        T: int,
        past_len: int,
        pos: torch.Tensor,
        seg_pos: torch.Tensor,
        last_reset_idx: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a 4D additive attention mask [B,1,T,P+T] with per-timestep resets."""
        # Past keys absolute start index currently in cache for this layer
        abs_past_start = pos - past_len  # [B]
        # Past tokens to mask due to prior segment start across calls
        n_invalid = torch.clamp(seg_pos - abs_past_start, min=0)
        n_invalid = torch.clamp(n_invalid, max=past_len)

        K = past_len + T
        k_idx = torch.arange(K, device=device).view(1, 1, -1).expand(B, T, -1)
        q_idx = torch.arange(T, device=device).view(1, -1)

        # Earliest allowed key index per (b,t)
        earliest_past = n_invalid.view(B, 1).expand(B, T)  # base if no reset yet
        earliest_from_resets = past_len + last_reset_idx  # P + r, with r=-1 if no reset
        earliest_allowed = torch.where(last_reset_idx >= 0, earliest_from_resets, earliest_past)

        # Latest allowed key index per (b,t) from causal structure
        latest_allowed = past_len + q_idx.expand(B, T)

        allowed = (k_idx >= earliest_allowed.unsqueeze(-1)) & (k_idx <= latest_allowed.unsqueeze(-1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        mask = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
        return mask.unsqueeze(1)  # [B,1,T,K]

    # ---- forward ----
    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        is_step = x.dim() == 2
        x_seq = x.unsqueeze(1) if is_step else x  # [B,1,H] or [B,T,H]
        B, T, H = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        # Ensure state
        if state is None or not isinstance(state, TensorDict) or state.batch_size == [] or (
            state.batch_size and state.batch_size[0] != B
        ):
            state = self.init_state(batch=B, device=device, dtype=dtype)

        # Normalize resets to [B,T]
        resets_bt = self._normalize_resets(resets, B, T, device)

        cache = state.get("cache")
        cache = self._unwrap_cache(cache)

        # Positions and per-batch segment starts
        pos = state.get("pos") if "pos" in state.keys() else torch.zeros(B, dtype=torch.long, device=device)
        seg_pos = state.get("seg_pos") if "seg_pos" in state.keys() else pos.clone()

        # Past length for this specific layer from cache
        layer_idx = int(getattr(self.hf_layer.self_attn, "layer_idx", 0))
        past_len = self._layer_past_len(cache, layer_idx)

        # Per-timestep last reset indices and additive mask implementing resets and causality
        last_reset_idx = self._build_last_reset_indices(resets_bt)
        attention_mask = self._build_additive_mask(
            B=B,
            T=T,
            past_len=past_len,
            pos=pos,
            seg_pos=seg_pos,
            last_reset_idx=last_reset_idx,
            dtype=dtype,
            device=device,
        )

        # Position ids account for segment resets: continue counting since last segment start
        base_since_seg = (pos - seg_pos).view(B, 1).expand(B, T)
        ar = torch.arange(T, device=device).view(1, T).expand(B, T)
        position_ids = torch.where(last_reset_idx >= 0, ar - last_reset_idx, base_since_seg + ar)
        cache_position = torch.arange(int(pos.min().item()), int(pos.min().item()) + T, device=device)
        position_embeddings = self.hf_submodel.rotary_emb(x_seq, position_ids)

        # Call the HF layer directly
        y = self.hf_layer(
            hidden_states=x_seq,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=True,
        )

        # Maintain rolling window if requested
        if self.mem_len > 0:
            cache.crop(self.mem_len)

        new_pos = pos + T
        # Persist last segment start across calls (absolute index of the last reset within this chunk, if any)
        last_reset_in_chunk = last_reset_idx[:, -1]
        new_seg_pos = torch.where(last_reset_in_chunk >= 0, pos + last_reset_in_chunk, seg_pos)
        out_state = TensorDict({"cache": cache, "pos": new_pos, "seg_pos": new_seg_pos}, batch_size=[B])
        update_parent_state(out_state, state)

        y_out: Tensor = y.squeeze(1) if is_step else y
        return y_out, out_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Reset the entire state for the current batch; ignore mask and mirror init_state."""
        if state is None:
            return None
        B = state.batch_size[0] if state.batch_size else 1
        device = state["pos"].device if "pos" in state.keys() else torch.device("cpu")
        new = self.init_state(batch=B, device=device, dtype=torch.float32)
        update_parent_state(new, state)
        return new

__all__ = ["HFLlamaLayerConfig", "HFLlamaLayerCell"]
