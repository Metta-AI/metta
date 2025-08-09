"""TorchRL-compatible PPO loss that re-uses Metta's custom advantage pipeline.

This module provides a minimal shim between TorchRL's ``ClipPPOLoss`` API and
Metta's existing advantage / PER logic implemented in :pymod:`metta.rl.functions`.
It lets us adopt TorchRL's objective signature without losing the CUDA-powered
`compute_puff_advantage` kernel or prioritized-replay weighting.

If *torchrl* is not available at runtime the class degrades gracefully to a
no-op base class so that the rest of the Metta codebase can still be imported
for downstream tooling (e.g. docs generation) that does not require training.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from metta.rl.functions import (
    compute_advantage,
    normalize_advantage_distributed,
)

try:
    # TorchRL is an optional dependency at the moment.
    from torchrl.objectives import ClipPPOLoss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback for doc builds

    class _ClipPPOLossFallback:  # pylint: disable=too-few-public-methods
        """Shallow fallback that mimics the interface without computing anything."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

        def forward(self, data: Dict[str, Tensor]):  # noqa: D401
            # Return zero losses so callers that expect keys can proceed.
            device = next(iter(data.values())).device if data else torch.device("cpu")
            zero = torch.zeros((), device=device)
            return {
                "loss_objective": zero,
                "loss_critic": zero,
                "loss_entropy": zero,
            }

    ClipPPOLoss = _ClipPPOLossFallback  # type: ignore  # noqa: N816


class MettaPPOLoss(ClipPPOLoss):  # type: ignore[misc]
    """Drop-in replacement for :class:`torchrl.objectives.ClipPPOLoss`.

    The only deviation from the base class is that we recompute the
    *advantages* using Metta's custom CUDA kernel that implements V-trace,
    prioritized-experience replay weights, and advantage normalization.  All
    other bookkeeping (entropy, ratio, clipping, etc.) is delegated back to the
    parent implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        # TorchRL's ClipPPOLoss consumes ``actor_network``/``critic_network`` plus
        # various hyper-parameters. We transparently forward everything.
        super().__init__(*args, **kwargs)

        # Cache hyper-parameters we need for Metta advantage function. We fall
        # back to sane defaults if the caller did not specify them.
        self._gamma: float = kwargs.get("gamma", 0.99)
        self._gae_lambda: float = kwargs.get("gae_lambda", 0.95)
        self._rho_clip: float = kwargs.get("vtrace_rho_clip", 1.0)
        self._c_clip: float = kwargs.get("vtrace_c_clip", 1.0)

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _compute_metta_advantage(self, td: Dict[str, Tensor]) -> Tensor:
        """Return (optionally normalized) advantage tensor inplace."""
        adv = compute_advantage(
            td["values"],
            td["rewards"],
            td["dones"],
            td["importance_sampling_ratio"],
            td["advantages"],
            self._gamma,
            self._gae_lambda,
            self._rho_clip,
            self._c_clip,
            device=td["values"].device,
        )
        adv = normalize_advantage_distributed(adv, True)
        return adv

    # ------------------------------------------------------------------
    # Public interface override
    # ------------------------------------------------------------------
    def forward(self, data: Dict[str, Tensor], *args: Any, **kwargs: Any):  # noqa: D401
        """Compute PPO losses on *data* after refreshing Metta advantages."""
        if "advantages" in data:
            data["advantages"] = self._compute_metta_advantage(data)
        return super().forward(data, *args, **kwargs)


__all__ = ["MettaPPOLoss"]
