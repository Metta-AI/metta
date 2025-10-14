from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.config import AxonsConfig
from cortex.types import MaybeState, ResetMask, Tensor

from .axon_cell import AxonCell


class AxonLayer(nn.Module):
    """Stateful replacement for ``nn.Linear`` backed by ``AxonCell``.

    - Accepts any input/output feature sizes. Internally sets
      ``hidden_size = in_features`` and ``out_dim = out_features`` on the
      wrapped ``AxonCell``.
    - Manages Axons state automatically, either inside a provided parent
      ``TensorDict`` under a configurable group/key, or in a local internal
      state when no parent state is supplied.
    - Respects both step inputs ``[B, H_in]`` and sequence inputs
      ``[B, T, H_in]``. Resets are forwarded transparently.
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
            # Heuristic rank: half of the smaller IO capacity
            cfg.out_rank = max(1, min(self.out_features, 2 * self.in_features) // 2)
            cfg.cuda_seq_threshold = 1000
        else:
            # Enforce IO sizes regardless of provided cfg
            cfg.hidden_size = self.in_features
            cfg.out_dim = self.out_features
            if cfg.out_rank is None:
                cfg.out_rank = max(1, min(self.out_features, 2 * self.in_features) // 2)

        # Wrapped AxonCell (allow out_dim != hidden_size)
        self.cell = AxonCell(cfg, enforce_out_dim_eq_hidden=False)

        # Local state used when parent state is not provided
        self._local_state: MaybeState = None

    # Expose state path for debugging/tests
    def state_path(self) -> Tuple[str, str]:
        return self._state_group, self._state_key

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype, state: MaybeState) -> MaybeState:
        """Ensure a substate exists either in the provided parent or locally."""
        if state is None:
            if self._local_state is None:
                self._local_state = self.cell.init_state(batch=batch, device=device, dtype=dtype)
            else:
                # Re-init on batch change
                if (self._local_state.batch_size and self._local_state.batch_size[0] != batch) or (
                    self._local_state["hc1"].device != device or self._local_state["hc1"].dtype != dtype
                ):
                    self._local_state = self.cell.init_state(batch=batch, device=device, dtype=dtype)
            return self._local_state

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
        if st is self._local_state:
            y, self._local_state = self.cell(x, self._local_state, resets=resets)
        else:
            group_td = st.get(self._state_group)
            assert group_td is not None
            sub = group_td.get(self._state_key)
            y, sub_new = self.cell(x, sub, resets=resets)
            group_td[self._state_key] = sub_new

        return y

    @torch.no_grad()
    def reset_state(self, mask: ResetMask, state: MaybeState | None = None) -> MaybeState | None:
        """Apply resets to the managed AxonCell state.

        If a parent ``state`` is provided, resets are applied in-place to the
        registered substate and the parent is returned. Otherwise, the local
        internal state is reset.
        """
        if state is None:
            if self._local_state is None:
                return None
            self._local_state = self.cell.reset_state(self._local_state, mask)  # type: ignore[arg-type]
            return None

        if self._state_group in state.keys():
            group_td = state.get(self._state_group)
            if group_td is not None and self._state_key in group_td.keys():
                group_td[self._state_key] = self.cell.reset_state(group_td[self._state_key], mask)  # type: ignore[arg-type]
        return state


__all__ = ["AxonLayer"]
