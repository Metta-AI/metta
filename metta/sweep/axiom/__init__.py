"""tAXIOM: tiny AXIOM - A hyperlightweight pipeline DSL for experimentation."""

from metta.sweep.axiom.checks import FAIL, WARN, grad_band, in_range, no_nan, prob_simplex, required_keys
from metta.sweep.axiom.core import Context, Ctx, Pipeline, PipelineState, Stage
from metta.sweep.axiom.hooks import Hook
from metta.sweep.axiom.manifest import diff_manifests, format_manifest_diff, summarize_manifest
from metta.sweep.axiom.sequential_sweep import SequentialSweepPipeline

__all__ = [
    # Core Pipeline DSL
    "Pipeline",
    "PipelineState",
    "Context",
    "Stage",
    "Ctx",  # Legacy, deprecated
    "Hook",
    # Canonical Pattern
    "SequentialSweepPipeline",
    # Manifest utilities
    "diff_manifests",
    "format_manifest_diff",
    "summarize_manifest",
    # Checks for validation
    "WARN",
    "FAIL",
    "required_keys",
    "no_nan",
    "prob_simplex",
    "grad_band",
    "in_range",
]