"""Hartmann6 optimization using PROTEIN with tAXIOM.

This example demonstrates using tAXIOM to orchestrate PROTEIN optimization
of the 6-dimensional Hartmann function with phase scheduling.
"""

import time
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from metta.sweep.axiom import Ctx, Pipeline, context_aware
from metta.sweep.protein import Protein

# ============================================================================
# Domain Models
# ============================================================================


class Phase(BaseModel):
    """Optimization phase configuration."""

    name: str
    expansion_rate: float = 0.25
    suggestions_per_pareto: int = 64


class Trial(BaseModel):
    """Single optimization trial."""

    id: int
    params: Dict[str, float]
    score: Optional[float] = None
    phase: Optional[str] = None
    elapsed: float = 0.0


class Summary(BaseModel):
    """Experiment summary."""

    trials: List[Trial]
    best_score: float
    best_params: Dict[str, float]
    total_time: float

    def by_phase(self) -> Dict[str, List[Trial]]:
        """Group trials by phase."""
        result = {}
        for trial in self.trials:
            if trial.phase:
                result.setdefault(trial.phase, []).append(trial)
        return result


class Analysis(BaseModel):
    """Experiment analysis."""

    best_score: float
    gap_to_optimum: float
    relative_error: float
    phase_performance: Dict[str, float]


# ============================================================================
# Hartmann6 Test Function
# ============================================================================


def hartmann6(x: np.ndarray) -> float:
    """6D Hartmann function. Global minimum: -3.32237."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    outer_sum = 0
    for i in range(4):
        inner_sum = sum(A[i, j] * (x[j] - P[i, j]) ** 2 for j in range(6))
        outer_sum += alpha[i] * np.exp(-inner_sum)

    return -outer_sum


# ============================================================================
# Pipeline Stages
# ============================================================================


@context_aware
def choose_phase(ctx: Ctx) -> Phase:
    """Select optimization phase based on trial count."""
    trial_id = ctx.metadata.get("trial_id", 0)

    # Simple 3-phase schedule with expansion rate for naive acquisition
    if trial_id < 30:
        return Phase(name="explore", expansion_rate=0.5, suggestions_per_pareto=128)
    elif trial_id < 50:
        return Phase(name="balance", expansion_rate=0.25, suggestions_per_pareto=64)
    else:
        return Phase(name="exploit", expansion_rate=0.1, suggestions_per_pareto=32)


@context_aware
def load_optimizer(ctx: Ctx) -> Protein:
    """Load or create PROTEIN optimizer."""
    # Get phase from previous stage output
    phase = ctx.get_stage_output("choose_phase")
    if phase is None:
        phase = Phase(name="balance", ucb_beta=2.0, expansion_rate=0.25)

    # Try to get existing optimizer
    optimizer = ctx.metadata.get("optimizer")

    if optimizer is None:
        # Create new optimizer
        sweep_config = {
            "metric": "hartmann6",
            "goal": "minimize",
        }

        # Add parameter spaces
        for i in range(6):
            sweep_config[f"x{i}"] = {
                "distribution": "uniform",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "scale": "auto",
            }

        optimizer = Protein(
            sweep_config=sweep_config,
            acquisition_fn="naive",  # Using PROTEIN's original acquisition function
            expansion_rate=phase.expansion_rate,
            suggestions_per_pareto=phase.suggestions_per_pareto,
        )

        # Replay observations if any
        trials_data = ctx.metadata.get("trials", [])
        for trial_dict in trials_data:
            if trial_dict.get("score") is not None:
                optimizer.observe(trial_dict["params"], trial_dict["score"], cost=1.0)
    else:
        # Update existing optimizer with new phase params
        optimizer.expansion_rate = phase.expansion_rate
        optimizer.suggestions_per_pareto = phase.suggestions_per_pareto

    # Store optimizer back
    ctx.metadata["optimizer"] = optimizer
    return optimizer


def suggest(optimizer: Protein) -> Dict[str, float]:
    """Get next suggestion from optimizer."""
    params, _ = optimizer.suggest()
    return params


@context_aware
def evaluate_with_context(ctx: Ctx) -> Trial:
    """Evaluate parameters and create trial."""
    # Get params from previous stage
    params = ctx.get_stage_output("suggest")
    if params is None:
        params = {f"x{i}": 0.5 for i in range(6)}

    start = time.time()

    # Evaluate function
    x = np.array([params[f"x{i}"] for i in range(6)])
    score = hartmann6(x)

    # Get current phase
    phase = ctx.get_stage_output("choose_phase")

    # Create trial
    trial = Trial(
        id=ctx.metadata.get("trial_id", 0),
        params=params,
        score=score,
        phase=phase.name if phase else "unknown",
        elapsed=time.time() - start,
    )

    # Update optimizer
    optimizer = ctx.metadata.get("optimizer")
    if optimizer:
        optimizer.observe(params, score, cost=1.0)

    # Store trial as dict for serialization
    trials = ctx.metadata.get("trials", [])
    trials.append(trial.dict())
    ctx.metadata["trials"] = trials

    # Update best
    best_score = ctx.metadata.get("best_score", float("inf"))
    if score < best_score:
        ctx.metadata["best_score"] = score
        ctx.metadata["best_params"] = params

    # Increment trial counter
    ctx.metadata["trial_id"] = trial.id + 1

    return trial


@context_aware
def summarize(ctx: Ctx) -> Summary:
    """Create experiment summary."""
    trials_data = ctx.metadata.get("trials", [])
    # Convert dicts back to Trial objects
    trials = [Trial(**t) for t in trials_data]
    best_score = ctx.metadata.get("best_score", float("inf"))
    best_params = ctx.metadata.get("best_params", {})
    start_time = ctx.metadata.get("start_time", time.time())

    return Summary(trials=trials, best_score=best_score, best_params=best_params, total_time=time.time() - start_time)


def analyze(summary: Summary) -> Analysis:
    """Analyze experiment results."""
    optimum = -3.32237

    # Calculate phase performance
    phase_performance = {}
    for phase_name, phase_trials in summary.by_phase().items():
        if phase_trials:
            best_in_phase = min(t.score for t in phase_trials if t.score is not None)
            phase_performance[phase_name] = best_in_phase

    return Analysis(
        best_score=summary.best_score,
        gap_to_optimum=abs(summary.best_score - optimum),
        relative_error=abs(summary.best_score - optimum) / abs(optimum),
        phase_performance=phase_performance,
    )


def report(analysis: Analysis) -> None:
    """Print experiment report."""
    print("\n" + "=" * 60)
    print("Hartmann6 Optimization Results")
    print("=" * 60)
    print(f"Best score:     {analysis.best_score:.5f}")
    print("Global optimum: -3.32237")
    print(f"Gap:            {analysis.gap_to_optimum:.5f}")
    print(f"Relative error: {analysis.relative_error:.2%}")

    if analysis.phase_performance:
        print("\nPerformance by phase:")
        for phase, score in analysis.phase_performance.items():
            print(f"  {phase:10s}: {score:.5f}")

    # Success message
    if analysis.gap_to_optimum < 0.01:
        print("\n✓ Excellent! Within 0.01 of optimum")
    elif analysis.gap_to_optimum < 0.1:
        print("\n✓ Good! Within 0.1 of optimum")
    else:
        print(f"\n⚠ Gap of {analysis.gap_to_optimum:.3f} from optimum")


# ============================================================================
# Pipeline Construction
# ============================================================================

"""
Data Flow Through Pipeline:

1. Trial Pipeline (single optimization iteration):
   ∅ → choose_phase → Phase
   Phase → load_optimizer → Protein
   Protein → suggest → Dict[str, float]
   Dict[str, float] → evaluate → Trial

2. Main Pipeline (full experiment):
   ∅ → run_trials → Summary
   Summary → analyze → Analysis  
   Analysis → report → None
"""


def build_pipeline() -> Pipeline:
    """Build the optimization pipeline with explicit data contracts."""

    # Single trial pipeline with data contracts
    trial_pipeline = (
        Pipeline()
        .stage("choose_phase", choose_phase)
        .through(Phase)  # Returns Phase configuration
        .stage("load_optimizer", load_optimizer)
        .through(Protein)  # Uses Phase from context -> Protein optimizer
        .stage("suggest", suggest)
        .through(Dict[str, float], input_type=Protein)  # Protein -> params dict
        .stage("evaluate", evaluate_with_context)
        .through(Trial)  # Uses params from context -> Trial
    )

    # Main pipeline with data contracts
    main_pipeline = (
        Pipeline()
        .stage("run_trials", lambda: run_trials(trial_pipeline, n_trials=70))
        .through(Summary)  # Returns Summary of all trials
        .stage("analyze", analyze)
        .through(Analysis, input_type=Summary)  # Summary -> Analysis
        .stage("report", report)
        .through(type(None), input_type=Analysis)  # Analysis -> None (prints)
    )

    return main_pipeline


def run_trials(pipeline: Pipeline, n_trials: int) -> Summary:
    """Run optimization trials."""
    ctx = Ctx()
    ctx.metadata["trial_id"] = 0
    ctx.metadata["trials"] = []
    ctx.metadata["start_time"] = time.time()

    for i in range(n_trials):
        ctx.metadata["trial_id"] = i
        pipeline.run(ctx)

        # Progress update
        if (i + 1) % 10 == 0:
            best = ctx.metadata.get("best_score", float("inf"))
            print(f"Trial {i + 1:3d}/{n_trials}: best = {best:.5f}")

    return summarize(ctx)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run Hartmann6 optimization with PROTEIN + tAXIOM."""
    print("Hartmann6 Optimization with PROTEIN + tAXIOM")
    print("=" * 60)

    pipeline = build_pipeline()
    pipeline.run()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
