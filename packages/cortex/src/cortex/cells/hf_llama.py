"""LLaMA decoder-layer wrapper that calls the HF layer and manages Static KV cache via TensorDict."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn
from tensordict import TensorDict  # type: ignore[import-untyped]

from cortex.cells.base import MemoryCell
from cortex.cells.core import update_parent_state
from cortex.cells.registry import register_cell
from cortex.config import CellConfig
from cortex.types import MaybeState, ResetMask, Tensor


class LayerwiseCache:
    """Per-layer sliding-window cache backed by preallocated K/V tensors."""

    def __init__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        mem_len: int,
        layer_idx: int,
        kv_len: int,
    ) -> None:
        self.keys = k
        self.values = v
        self.mem_len = int(mem_len)
        self.layer_idx = int(layer_idx)
        self.kv_len = int(kv_len)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = layer_idx
        _ = cache_kwargs

        B, n_kv, T, _ = key_states.shape
        past_len = int(self.kv_len)

        if past_len > 0:
            k_past = self.keys[:, :, :past_len, :]
            v_past = self.values[:, :, :past_len, :]
            k_full = torch.cat((k_past, key_states), dim=2)
            v_full = torch.cat((v_past, value_states), dim=2)
        else:
            k_full = key_states
            v_full = value_states
        total_len = past_len + T
        new_past = min(total_len, self.mem_len)
        if new_past > 0:
            window_k = k_full[:, :, -new_past:, :].contiguous()
            window_v = v_full[:, :, -new_past:, :].contiguous()
            self.keys.zero_()
            self.values.zero_()
            self.keys[:, :, :new_past, :] = window_k
            self.values[:, :, :new_past, :] = window_v
        else:
            self.keys.zero_()
            self.values.zero_()

        self.kv_len = new_past
        return k_full, v_full


class HFLlamaLayerConfig(CellConfig):
    """Config for the HF LLaMA layer wrapper; expects hf_layer, hf_submodel, hf_config set at runtime."""

    cell_type: str = "hf_llama_layer"

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
        hf_layer = getattr(cfg, "hf_layer", None)
        hf_submodel = getattr(cfg, "hf_submodel", None)
        hf_config = getattr(cfg, "hf_config", None)
        if not isinstance(hf_layer, nn.Module):
            raise ValueError("HFLlamaLayerConfig must include an 'hf_layer' nn.Module")
        if not isinstance(hf_submodel, nn.Module):
            raise ValueError("HFLlamaLayerConfig must include an 'hf_submodel' nn.Module (e.g., model.model)")
        self.hf_layer: nn.Module = hf_layer
        rotary_emb = getattr(hf_submodel, "rotary_emb", None)
        if rotary_emb is None:
            raise ValueError("hf_submodel is missing 'rotary_emb'")
        self.hf_rotary_emb: nn.Module = rotary_emb
        self.hf_config = hf_config

        self.mem_len = int(cfg.mem_len)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        if self.mem_len <= 0:
            pos = torch.zeros(batch, dtype=torch.long, device=device)
            seg_pos = torch.zeros(batch, dtype=torch.long, device=device)
            kv_len = torch.zeros(batch, dtype=torch.long, device=device)
            return TensorDict({"pos": pos, "seg_pos": seg_pos, "kv_len": kv_len}, batch_size=[batch])

        n_kv = int(getattr(self.hf_config, "num_key_value_heads", getattr(self.hf_config, "num_attention_heads", 1)))
        head_dim = int(
            getattr(
                self.hf_layer.self_attn,
                "head_dim",
                self.hidden_size // int(getattr(self.hf_config, "num_attention_heads", 1)),
            )
        )

        k = torch.zeros(batch, n_kv, self.mem_len, head_dim, dtype=dtype, device=device)
        v = torch.zeros_like(k)
        kv_len = torch.zeros(batch, dtype=torch.long, device=device)
        pos = torch.zeros(batch, dtype=torch.long, device=device)
        seg_pos = torch.zeros(batch, dtype=torch.long, device=device)
        return TensorDict({"k": k, "v": v, "kv_len": kv_len, "pos": pos, "seg_pos": seg_pos}, batch_size=[batch])

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

    def _prepare_layerwise_cache(
        self,
        state: TensorDict,
        *,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[LayerwiseCache, int, torch.Tensor]:
        """Create cache and return (cache, layer_idx, kv_len)."""
        if "k" not in state.keys() or "v" not in state.keys():
            state.update(self.init_state(batch=B, device=device, dtype=dtype))

        k = state.get("k")
        v = state.get("v")
        if k.shape[0] != B:
            raise ValueError(f"LayerwiseCache expects batch size {B}, got {k.shape[0]}")

        kv_len = state.get("kv_len") if "kv_len" in state.keys() else torch.zeros(B, dtype=torch.long, device=device)
        layer_idx = int(getattr(self.hf_layer.self_attn, "layer_idx", 0))
        kv_len_scalar = int(kv_len[0].item()) if kv_len.numel() > 0 else 0
        cache = LayerwiseCache(k=k, v=v, mem_len=self.mem_len, layer_idx=layer_idx, kv_len=kv_len_scalar)
        return cache, layer_idx, kv_len

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
        """Return additive mask [B,1,T,(P+T)] for dynamic semantics."""
        abs_past_start = pos - past_len
        n_invalid = torch.clamp(seg_pos - abs_past_start, min=0)
        n_invalid = torch.clamp(n_invalid, max=past_len)

        K = past_len + T
        k_idx = torch.arange(K, device=device).view(1, 1, -1).expand(B, T, -1)
        q_idx = torch.arange(T, device=device).view(1, -1)

        earliest_past = n_invalid.view(B, 1).expand(B, T)
        earliest_from_resets = past_len + last_reset_idx
        earliest_allowed = torch.where(last_reset_idx >= 0, earliest_from_resets, earliest_past)

        latest_allowed = past_len + q_idx.expand(B, T)

        allowed = (k_idx >= earliest_allowed.unsqueeze(-1)) & (k_idx <= latest_allowed.unsqueeze(-1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        mask = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
        return mask.unsqueeze(1)

    def _build_additive_mask_static(
        self,
        *,
        B: int,
        T: int,
        mem_len: int,
        kv_len: torch.Tensor,
        pos: torch.Tensor,
        seg_pos: torch.Tensor,
        last_reset_idx: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return additive mask [B,1,T,mem_len] implementing per‑timestep resets for static cache."""
        kv_len = kv_len.to(device=device)
        abs_past_start = pos - kv_len
        n_invalid = torch.clamp(seg_pos - abs_past_start, min=0)
        n_invalid = torch.clamp(n_invalid, max=kv_len)

        K = int(mem_len)
        k_idx = torch.arange(K, device=device).view(1, 1, -1).expand(B, T, -1)
        q_idx = torch.arange(T, device=device).view(1, -1).expand(B, T)

        kv_len_bt = kv_len.view(B, 1).expand(B, T)
        earliest_past = n_invalid.view(B, 1).expand(B, T)
        earliest_from_resets = kv_len_bt + last_reset_idx
        earliest_allowed = torch.where(last_reset_idx >= 0, earliest_from_resets, earliest_past)

        latest_allowed = kv_len_bt + q_idx
        latest_allowed = torch.clamp(latest_allowed, max=K - 1)
        earliest_allowed = torch.clamp(earliest_allowed, min=0)

        allowed = (k_idx >= earliest_allowed.unsqueeze(-1)) & (k_idx <= latest_allowed.unsqueeze(-1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        mask = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
        return mask.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        is_step = x.dim() == 2
        x_seq = x.unsqueeze(1) if is_step else x
        B, T, H = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if (
            state is None
            or not isinstance(state, TensorDict)
            or state.batch_size == []
            or (state.batch_size and state.batch_size[0] != B)
        ):
            state = self.init_state(batch=B, device=device, dtype=dtype)

        resets_bt = self._normalize_resets(resets, B, T, device)

        pos = state.get("pos") if "pos" in state.keys() else torch.zeros(B, dtype=torch.long, device=device)
        seg_pos = state.get("seg_pos") if "seg_pos" in state.keys() else pos.clone()

        last_reset_idx = self._build_last_reset_indices(resets_bt)

        attn_impl = getattr(self.hf_config, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = getattr(self.hf_config, "attn_implementation", None)
        use_flash = attn_impl == "flash_attention_2"

        base_since_seg = (pos - seg_pos).view(B, 1).expand(B, T)
        ar = torch.arange(T, device=device).view(1, T).expand(B, T)
        if use_flash and T > 1:
            position_ids = base_since_seg + ar
        else:
            position_ids = torch.where(last_reset_idx >= 0, ar - last_reset_idx, base_since_seg + ar)
        position_embeddings = self.hf_rotary_emb(x_seq, position_ids)

        if self.mem_len > 0:
            td_state = cast(TensorDict, state)
            kv_len = td_state.get("kv_len") if "kv_len" in td_state else torch.zeros(B, dtype=torch.long, device=device)
            cache, layer_idx, kv_len = self._prepare_layerwise_cache(state, B=B, T=T, dtype=dtype, device=device)
            past_len = int(kv_len[0].item()) if kv_len.numel() > 0 else 0
            if use_flash:
                if T > 1 and resets is not None and torch.count_nonzero(resets) > 0:
                    warnings.warn(
                        "FlashAttention: ignoring per‑timestep resets within chunk when T>1; "
                        "use bptt_horizon=1 or eager attention to enforce mid‑chunk resets.",
                        stacklevel=1,
                    )
                y = self.hf_layer(
                    hidden_states=x_seq,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                )
            else:
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
                y = self.hf_layer(
                    hidden_states=x_seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                )
            state.set("k", cache.keys)
            state.set("v", cache.values)
            new_kv_len = torch.full_like(kv_len, cache.kv_len)
            state.set("kv_len", new_kv_len)
        else:
            if use_flash:
                if T > 1 and resets is not None and torch.count_nonzero(resets) > 0:
                    warnings.warn(
                        "FlashAttention: ignoring per‑timestep resets within chunk when T>1; "
                        "use bptt_horizon=1 or eager attention to enforce mid‑chunk resets.",
                        stacklevel=1,
                    )
                y = self.hf_layer(
                    hidden_states=x_seq,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
            else:
                attention_mask = self._build_additive_mask(
                    B=B,
                    T=T,
                    past_len=0,
                    pos=pos,
                    seg_pos=seg_pos,
                    last_reset_idx=last_reset_idx,
                    dtype=dtype,
                    device=device,
                )
                y = self.hf_layer(
                    hidden_states=x_seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )

        new_pos = pos + T
        last_reset_in_chunk = last_reset_idx[:, -1]
        new_seg_pos = torch.where(last_reset_in_chunk >= 0, pos + last_reset_in_chunk, seg_pos)
        if self.mem_len > 0:
            out_state = TensorDict(
                {
                    "k": state.get("k"),
                    "v": state.get("v"),
                    "kv_len": state.get("kv_len"),
                    "pos": new_pos,
                    "seg_pos": new_seg_pos,
                },
                batch_size=[B],
            )
        else:
            out_state = TensorDict(
                {
                    "pos": new_pos,
                    "seg_pos": new_seg_pos,
                    "kv_len": torch.zeros(B, dtype=torch.long, device=device),
                },
                batch_size=[B],
            )
        update_parent_state(out_state, state)

        y_out: Tensor = y.squeeze(1) if is_step else y
        return y_out, out_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Mirror init_state behavior; ignore mask and reinitialize."""
        if state is None:
            return None
        B = state.batch_size[0] if isinstance(state, TensorDict) and state.batch_size else 1
        device = (
            state.get("pos").device if isinstance(state, TensorDict) and "pos" in state.keys() else torch.device("cpu")
        )
        new = self.init_state(batch=B, device=device, dtype=torch.float32)
        update_parent_state(new, state)
        return new


__all__ = ["HFLlamaLayerConfig", "HFLlamaLayerCell"]
