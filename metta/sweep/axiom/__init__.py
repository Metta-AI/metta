"""tAXIOM: tiny AXIOM - A minimal DSL for experiment orchestration in RL."""

# Import control_flow to add methods to Pipeline
import metta.sweep.axiom.control_flow  # noqa: F401
from metta.sweep.axiom.checks import FAIL, WARN, grad_band, in_range, no_nan, prob_simplex, required_keys
from metta.sweep.axiom.core import Ctx, Pipeline, Stage
from metta.sweep.axiom.hooks import Hook
from metta.sweep.axiom.sequential_sweep import SequentialSweep

__all__ = [
    "Pipeline",
    "Stage",
    "Ctx",
    "Hook",
    "SequentialSweep",
    # Checks
    "WARN",
    "FAIL",
    "required_keys",
    "no_nan",
    "prob_simplex",
    "grad_band",
    "in_range",
]
