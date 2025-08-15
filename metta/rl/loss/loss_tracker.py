from __future__ import annotations

from typing import Iterable


class LossTracker:
    """
    Flexible dynamic loss/metric tracker. Interacts heavily with BaseLoss and leans on implemented losses.
    TODO: investigate if this programmatic path has SPS implications. If so, delete LossTracker and have losses handle
    their own summing and zeroing. That would skip dict lookup.
    """

    # Metrics that should not be averaged in stats(); copied as-is
    _NON_AVERAGED_KEYS: set[str] = {"explained_variance"}

    def __init__(self):
        # Dynamic configuration
        self._tracked_metric_keys: set[str] | None = None
        self._dynamic_metric_sums: dict[str, float] = {}

        # Global minibatch counter for averaging
        self.minibatches_processed = 0

        self.zero()

    def configure_from_losses(self, losses: Iterable[object]):
        """Configure which metrics to track dynamically by querying loss instances."""
        collected: set[str] = set()
        for loss in losses:
            keys: list[str] | tuple[str, ...] | None = None
            if hasattr(loss, "losses_to_track"):
                try:
                    keys = loss.losses_to_track()
                except Exception:
                    keys = None
            if keys:
                for k in keys:
                    if isinstance(k, str) and k:
                        collected.add(k)
        self._tracked_metric_keys = collected
        for k in self._tracked_metric_keys:
            self._dynamic_metric_sums.setdefault(k, 0.0)

    def add(self, key: str, value: float) -> None:
        """Accumulate a dynamic metric value."""
        self._dynamic_metric_sums[key] = self._dynamic_metric_sums.get(key, 0.0) + float(value)

    def set(self, key: str, value: float) -> None:
        """Set a dynamic metric value (overwrites any existing value)."""
        self._dynamic_metric_sums[key] = float(value)

    def get(self, key: str) -> float:
        return float(self._dynamic_metric_sums.get(key, 0.0))

    def zero(self):
        """Reset all dynamic metrics and the minibatch counter."""
        for k in list(self._dynamic_metric_sums.keys()):
            self._dynamic_metric_sums[k] = 0.0
        self.minibatches_processed = 0

    def stats(self) -> dict[str, float]:
        """Return metrics as averages over `minibatches_processed` unless non-averaged."""
        n = max(1, self.minibatches_processed)

        def _avg_or_raw(key: str) -> float:
            val = float(self._dynamic_metric_sums.get(key, 0.0))
            if key in self._NON_AVERAGED_KEYS:
                return val
            return val / n

        keys: set[str]
        if self._tracked_metric_keys is not None and len(self._tracked_metric_keys) > 0:
            keys = self._tracked_metric_keys
        else:
            keys = set(self._dynamic_metric_sums.keys())

        return {k: _avg_or_raw(k) for k in sorted(keys)}
