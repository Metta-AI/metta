"""Hardware-aware sweep factory using the existing SweepTool.

This module exposes a tiny factory function `hw_sweep` that returns a
configured `SweepTool` using the optimized Protein configuration for a given
GPU/node pair. It keeps the hardware-scaling experiment simple: launch one
SweepTool per (gpus, nodes) via the CLI.
"""

from __future__ import annotations

from experiments.adaptive.hardware_scaling.optimized_sweep_config import (
    create_optimized_protein_config,
)
from metta.sweep.protein_config import ProteinConfig
from metta.tools.sweep import SweepSchedulerType, SweepTool


def hw_sweep(
    *,
    gpus: int,
    nodes: int,
    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate_in_sweep",
    total_timesteps: int = 300_000_000,
    protein_metric_path: str = "evaluator/eval_sweep/score",
) -> SweepTool:
    """Create a SweepTool configured for a specific hardware pair.

    Parameters
    ----------
    gpus: int
        Number of GPUs per job.
    nodes: int
        Number of nodes per job.
    recipe_module: str
        Recipe module path for training/evaluation entrypoints.
    train_entrypoint: str
        Training entrypoint name within the recipe module.
    eval_entrypoint: str
        Evaluation entrypoint name within the recipe module.
    total_timesteps: int
        Total training timesteps per trial (applied via train overrides).
    protein_metric_path: str
        Metric path to optimize in the sweep observations.

    Returns
    -------
    SweepTool
        A SweepTool instance preconfigured with optimized Protein settings and
        hardware (gpus/nodes). Remaining knobs like `max_trials`, `batch_size`,
        `max_parallel_jobs`, `dispatcher_type`, and wandb config are intended to
        be provided via CLI overrides when invoking tools/run.py.
    """
    # Build an optimized ProteinConfig for this hardware
    protein_cfg: ProteinConfig = create_optimized_protein_config(
        gpus=gpus,
        nodes=nodes,
        num_agents=24,
        bptt_horizon=64,
        metric_path=protein_metric_path,
    )

    # Return a SweepTool with recipe, entrypoints, and hardware set.
    # Training timesteps are applied via train_overrides, while other sweep
    # orchestration knobs remain CLI-configurable.
    tool = SweepTool(
        protein_config=protein_cfg,
        recipe_module=recipe_module,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        # Use async-capped scheduler rather than batched to better reflect
        # single-stream FoM runs and avoid batch barriers
        scheduler_type=SweepSchedulerType.ASYNC_CAPPED,
        # Default to 4 parallel jobs for FoM sweeps
        max_parallel_jobs=4,
        gpus=gpus,
        nodes=nodes,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )
    return tool
