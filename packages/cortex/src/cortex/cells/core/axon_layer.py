from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.config import AxonsConfig
from cortex.types import MaybeState, ResetMask, Tensor

from .axon_cell import AxonCell


class AxonLayer(nn.Module):
    """Stateful replacement for ``nn.Linear`` blending Linear and Axon paths.

    Behavior
    - Computes ``z = (1 - alpha) * LN(Linear(x)) + alpha * LN(Axon(x))`` where
      ``alpha = sigmoid(a)`` and ``a`` is a learnable scalar initialized ``<< 0``
      so the layer is effectively linear at initialization.
    - The Axon branch uses an ``AxonCell`` with activation forced to ``identity``.
    - Both branches are normalized with ``LayerNorm(out_features)`` before the
      convex blend to keep scales comparable.

    Interface
    - Accepts any input/output feature sizes. Internally forces
      ``hidden_size = in_features`` and ``out_dim = out_features`` on the
      wrapped ``AxonCell``.
    - Manages Axon state automatically in a provided parent ``TensorDict``
      under ``group/name`` or keeps a local internal state when none is
      provided.
    - Supports step inputs ``[B, H_in]`` and sequence inputs ``[B, T, H_in]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: Optional[AxonsConfig] = None,
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

        # Build Axons config with enforced IO sizes
        if cfg is None:
            cfg = AxonsConfig(hidden_size=self.in_features, out_dim=self.out_features)
            # Sensible defaults for a "linear-like" replacement
            cfg.activation = "identity"
            # Enable SRHT only when feature dim is a power-of-two; else disable to avoid FWHT constraint.
            is_pow2 = (self.in_features & (self.in_features - 1)) == 0 and self.in_features > 0
            cfg.use_srht = bool(is_pow2)
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

        # Plain linear branch for linear-at-init behavior
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        # Residual/convex gate parameter: alpha = sigmoid(alpha_logit)

    # Expose state path for debugging/tests
    def state_path(self) -> Tuple[str, str]:
        return self._state_group, self._state_key

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype, state: MaybeState) -> TensorDict:
        """Ensure a substate exists either in the provided parent or locally."""
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
        return state

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

        # Prepare substate
        st = self._ensure_state(batch=B, device=x.device, dtype=x.dtype, state=state)

        # Route to underlying AxonCell and write-back updated substate
        group_td = st.get(self._state_group)
        assert group_td is not None
        sub = group_td.get(self._state_key)
        y_axon, sub_new = self.cell(x, sub, resets=resets)
        group_td[self._state_key] = sub_new

        # Compute linear branch directly on input (supports [B, H] or [B, T, H])
        y_lin = self.linear(x)

        y = y_lin + y_axon
        return y

    @torch.no_grad()
    def reset_state(self, mask: ResetMask, state: MaybeState | None = None) -> MaybeState | None:
        """Apply resets to the managed AxonCell state.

        If a parent ``state`` is provided, resets are applied in-place to the
        registered substate and the parent is returned. Otherwise, the local
        internal state is reset.
        """
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
