"""Evaluation entrypoints for the CVC recipe."""

from __future__ import annotations

import typing

import experiments.recipes.cvc.core
import metta.sim.simulation_config
import metta.tools.eval

__all__ = ["evaluate", "make_eval_suite"]


def evaluate(
    *,
    policy_uris: str | typing.Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str = "standard",
    subset: typing.Optional[list[str]] = None,
) -> metta.tools.eval.EvaluateTool:
    """Evaluate policies on the canonical CoGs vs Clips missions."""
    return experiments.recipes.cvc.core.evaluate(
        policy_uris=policy_uris,
        num_cogs=num_cogs,
        difficulty=difficulty,
        subset=subset,
    )


def make_eval_suite(
    *,
    num_cogs: int = 4,
    difficulty: str = "standard",
    subset: typing.Optional[list[str]] = None,
) -> list[metta.sim.simulation_config.SimulationConfig]:
    """Construct a suite of evaluation simulations."""
    return experiments.recipes.cvc.core.make_eval_suite(
        num_cogs=num_cogs,
        difficulty=difficulty,
        subset=subset,
    )
