from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.config import AxonConfig
from cortex.types import MaybeState, ResetMask, Tensor

from .axon_cell import AxonCell


class AxonLayer(nn.Module):
    """Stateful linear-like layer: y = Linear(x) + AxonCell(x); substate at state[group][name]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: Optional[AxonConfig] = None,
        *,
        name: Optional[str] = None,
        group: str = "axon",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._state_group = group
        # Stable key for storing substate when used inside a parent TensorDict
        self._state_key = name if name is not None else f"axonlayer_{id(self)}"

        # Build Axon config with enforced IO sizes
        if cfg is None:
            cfg = AxonConfig(hidden_size=self.in_features, out_dim=self.out_features)
            # Sensible defaults for a "linear-like" replacement
            cfg.activation = "identity"
            # Choose input mixing: if power-of-two dim, prefer SRHT; otherwise use untraced linear
            is_pow2 = (self.in_features & (self.in_features - 1)) == 0 and self.in_features > 0
            cfg.use_srht = bool(is_pow2)
            cfg.use_untraced_linear = not is_pow2
            cfg.srht_permute = True
            cfg.cuda_seq_threshold = 1000
        else:
            # Enforce IO sizes regardless of provided cfg
            cfg.hidden_size = self.in_features
            cfg.out_dim = self.out_features
            # Keep Axon activation linear inside this wrapper
            cfg.activation = "identity"

        # Wrapped AxonCell (allow out_dim != hidden_size)
        self.cell = AxonCell(cfg, enforce_out_dim_eq_hidden=False)

        # Plain linear branch
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        # No gating/normalization inside this wrapper

    # Expose state path for debugging/tests
    def state_path(self) -> Tuple[str, str]:
        return self._state_group, self._state_key

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype, state: MaybeState) -> TensorDict:
        """Ensure state[group][name] exists with correct batch/device/dtype and return the group TensorDict."""
        if state is None:
            raise ValueError("AxonLayer requires an explicit parent TensorDict state.")

        # Parent state path: create group/key lazily
        if self._state_group not in state.keys():
            state[self._state_group] = TensorDict({}, batch_size=[batch])
        group_td = state.get(self._state_group)
        assert group_td is not None
        if self._state_key not in group_td.keys():
            group_td[self._state_key] = self.cell.init_state(batch=batch, device=device, dtype=dtype)
        else:
            td = group_td[self._state_key]
            if (td.batch_size and td.batch_size[0] != batch) or (
                td["hc1"].device != device or td["hc1"].dtype != dtype
            ):
                group_td[self._state_key] = self.cell.init_state(batch=batch, device=device, dtype=dtype)
        # Return the group TensorDict so the caller can read/write the substate in-place.
        return group_td

    def forward(
        self,
        x: Tensor,
        state: MaybeState | None = None,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tensor:
        is_step = x.dim() == 2
        if is_step:
            B, H_in = x.shape
        else:
            B, _, H_in = x.shape
        assert H_in == self.in_features, f"AxonLayer expected input dim={self.in_features}, got {H_in}"

        # Prepare group TensorDict + substate
        group_td = self._ensure_state(batch=B, device=x.device, dtype=x.dtype, state=state)

        # Route to underlying AxonCell and write-back updated substate
        sub = group_td.get(self._state_key)
        y_axon, sub_new = self.cell(x, sub, resets=resets)
        group_td[self._state_key] = sub_new

        # Compute linear branch directly on input (supports [B, H] or [B, T, H])
        y_lin = self.linear(x)

        y = y_lin + y_axon
        return y

    @torch.no_grad()
    def reset_state(self, mask: ResetMask, state: MaybeState | None = None) -> MaybeState | None:
        """Reset AxonCell substate in-place on the parent TensorDict and return the parent."""
        if state is None:
            raise ValueError("AxonLayer.reset_state requires a parent TensorDict state.")

        if self._state_group in state.keys():
            group_td = state.get(self._state_group)
            if group_td is not None and self._state_key in group_td.keys():
                group_td[self._state_key] = self.cell.reset_state(group_td[self._state_key], mask)  # type: ignore[arg-type]
        return state


def update_parent_state(parent: TensorDict, source: TensorDict) -> TensorDict:
    """Merge auxiliary entries from ``source`` into ``parent`` without clobbering fresh values."""
    for key in source.keys():
        if key in parent.keys():
            continue
        parent[key] = source.get(key)
    return parent


__all__ = ["AxonLayer", "update_parent_state"]
