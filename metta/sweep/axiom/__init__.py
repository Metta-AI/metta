"""tAXIOM: tiny AXIOM - A minimal DSL for experiment orchestration in RL."""

# Import control_flow to add methods to Pipeline
# import metta.sweep.axiom.control_flow  # noqa: F401  # File removed during cleanup
from metta.sweep.axiom.checks import FAIL, WARN, grad_band, in_range, no_nan, prob_simplex, required_keys
from metta.sweep.axiom.core import Ctx, Pipeline, Stage
from metta.sweep.axiom.experiment import AxiomExperiment, RunHandle
from metta.sweep.axiom.experiment_spec import (
    AxiomControls,
    ComparisonExperimentSpec,
    ExperimentSpec,
    SweepExperimentSpec,
    TrainingExperimentSpec,
    load_experiment_spec,
    save_experiment_spec,
)
from metta.sweep.axiom.hooks import Hook
from metta.sweep.axiom.manifest import diff_manifests, format_manifest_diff, summarize_manifest
from metta.sweep.axiom.sequential_sweep import SequentialSweepPipeline

__all__ = [
    "Pipeline",
    "Stage",
    "Ctx",
    "Hook",
    "SequentialSweepPipeline",
    # Experiment
    "AxiomExperiment",
    "RunHandle",
    # Experiment Specs
    "ExperimentSpec",
    "TrainingExperimentSpec",
    "SweepExperimentSpec",
    "ComparisonExperimentSpec",
    "AxiomControls",
    "load_experiment_spec",
    "save_experiment_spec",
    # Manifest
    "diff_manifests",
    "format_manifest_diff",
    "summarize_manifest",
    # Checks
    "WARN",
    "FAIL",
    "required_keys",
    "no_nan",
    "prob_simplex",
    "grad_band",
    "in_range",
]
