"""Nevergrad optimizer adapter implementing the Optimizer protocol used by schedulers.

This adapter builds a nested Nevergrad parametrization from our canonical
ParameterSpec types and exposes a suggest(observations, n_suggestions) API.

Notes:
- Nevergrad minimizes a scalar loss; we convert score based on goal.
- We maintain an ask→tell mapping using frozen suggestion signatures since the
  scheduler does not carry Nevergrad Candidate objects in observations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from metta.sweep.core import CategoricalParameterConfig, ParameterConfig
from metta.sweep.nevergrad_config import NevergradConfig
from mettagrid.util.dict_utils import unroll_nested_dict

logger = logging.getLogger(__name__)


class NevergradOptimizer:
    """Adapter for Nevergrad ask/tell optimization."""

    def __init__(self, config: NevergradConfig):
        self.config = config
        # Lazily import nevergrad to avoid hard dependency at import time
        try:
            import nevergrad as ng  # type: ignore
        except Exception as e:  # pragma: no cover - import-time environment dependent
            raise ImportError("Nevergrad is required for NevergradOptimizer. Please install 'nevergrad'.") from e

        self._ng = ng
        self._parametrization = self._build_parametrization(config.parameters)

        # Select optimizer implementation
        optimizer_name = config.settings.optimizer_name
        budget = config.settings.budget
        num_workers = config.settings.num_workers
        try:
            opt_cls = getattr(self._ng.optimizers, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Unknown Nevergrad optimizer: {optimizer_name}") from e

        self._optimizer = opt_cls(parametrization=self._parametrization, budget=budget, num_workers=num_workers)
        if config.settings.seed is not None:
            try:
                self._optimizer._rng.seed(config.settings.seed)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Nevergrad optimizer does not expose _rng or seed; ignoring seed setting")

        # Map frozen suggestion signatures to Nevergrad Candidate objects
        self._asked: dict[Tuple[Tuple[str, Any], ...], Any] = {}
        self._told: set[Tuple[Tuple[str, Any], ...]] = set()

    # ------------------ Public API ------------------
    def suggest(self, observations: list[dict[str, Any]], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions via Nevergrad.

        - Tells the optimizer about newly completed observations.
        - Asks for n_suggestions new candidates and returns nested dicts.
        """
        # Tell new observations
        for obs in observations or []:
            nested = obs.get("suggestion", {})
            sig = _freeze_signature(nested)
            if sig in self._told:
                continue
            cand = self._asked.get(sig)
            if cand is None:
                # This can happen if scheduler restarts or rounding mismatch; skip tell.
                logger.debug("[NevergradOptimizer] Unrecognized observation signature; skipping tell")
                continue
            loss = _to_loss(score=float(obs.get("score", 0.0)), goal=self.config.goal)
            try:
                self._optimizer.tell(cand, loss)
                self._told.add(sig)
            except Exception as e:
                logger.warning(f"[NevergradOptimizer] tell() failed for signature {sig}: {e}")

        # Ask for new suggestions
        results: list[dict[str, Any]] = []
        for _ in range(max(0, int(n_suggestions))):
            cand = self._optimizer.ask()
            suggestion = cand.value  # Nested dict of parameter values
            sig = _freeze_signature(suggestion)
            self._asked[sig] = cand
            results.append(suggestion)

        return results

    # ------------------ Internal helpers ------------------
    def _build_parametrization(self, params: Dict[str, Any]):
        """Build a nested ng.p.Dict parametrization from our ParameterSpec dict."""
        p = self._ng.p

        def convert(node: Any):
            if isinstance(node, ParameterConfig):
                dist = node.distribution
                if dist == "uniform":
                    return p.Scalar(lower=float(node.min), upper=float(node.max))
                if dist == "int_uniform":
                    return p.Scalar(lower=int(node.min), upper=int(node.max)).set_integer_casting()
                if dist == "uniform_pow2":
                    # Build discrete choices of powers-of-two within [min, max]
                    from math import ceil, floor, log2

                    if node.min <= 0 or node.max <= 0:
                        raise ValueError("uniform_pow2 requires positive bounds")
                    lo_e = ceil(log2(node.min))
                    hi_e = floor(log2(node.max))
                    if lo_e > hi_e:
                        raise ValueError("No powers-of-two within [min, max] bounds")
                    values = [2**k for k in range(lo_e, hi_e + 1)]
                    return p.Choice(values)
                if dist == "log_normal":
                    # Use Nevergrad's dedicated log-scale positive parameter
                    return p.Log(lower=float(node.min), upper=float(node.max))
                if dist == "logit_normal":
                    eps = 1e-6
                    lo = max(eps, float(node.min))
                    hi = min(1 - eps, float(node.max))
                    return p.Scalar(lower=lo, upper=hi)
                raise ValueError(f"Unsupported distribution: {dist}")

            if isinstance(node, CategoricalParameterConfig):
                return p.Choice(list(node.choices))

            if isinstance(node, dict):
                # Recursively convert nested mapping
                converted = {k: convert(v) for k, v in node.items()}
                return p.Dict(**converted)

            # Static value or unknown type: represent as a fixed Choice with single value
            return p.Choice([node])

        return convert(params)


def _freeze_signature(nested: dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Create a stable signature for a nested suggestion dict.

    Uses unroll_nested_dict to flatten paths. Rounds floats for stability.
    """
    flat = dict(unroll_nested_dict(nested))
    items: list[tuple[str, Any]] = []
    for k, v in flat.items():
        if isinstance(v, float):
            v = round(v, 12)
        items.append((k, v))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def _to_loss(*, score: float, goal: str) -> float:
    return -score if goal == "maximize" else score
