#!/usr/bin/env python3
"""Launch one sweep per hardware pair by shelling out to tools/run.py.

Example:
  python experiments/adaptive/hardware_scaling/launch.py \
      --wandb-entity YOUR_ENTITY \
      --wandb-project hardware-scaling \
      --gpu-counts 1 2 4 8 \
      --node-counts 1 2 \
      --dispatcher skypilot \
      --max-trials 50 --batch-size 4 --max-parallel-jobs 4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import datetime
from typing import Iterable, List, Tuple

from metta.common.util.log_config import init_logging


logger = logging.getLogger(__name__)


def _pairs(
    gpu_counts: Iterable[int], node_counts: Iterable[int]
) -> List[Tuple[int, int]]:
    return [(g, n) for g in gpu_counts for n in node_counts if g <= 8 * n]


def _build_cmd(
    *,
    gpus: int,
    nodes: int,
    sweep_name: str,
    wandb_entity: str,
    wandb_project: str,
    dispatcher: str,
    max_trials: int,
    batch_size: int,
    max_parallel_jobs: int,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    total_timesteps: int,
    metric_path: str,
) -> list[str]:
    """Construct the CLI command to launch a single sweep via run_tool."""
    return [
        "uv",
        "run",
        "./tools/run.py",
        "experiments.adaptive.hardware_scaling.hw_sweep",
        # Function args (bound to factory parameters)
        f"gpus={gpus}",
        f"nodes={nodes}",
        f"recipe_module={recipe}",
        f"train_entrypoint={train_entrypoint}",
        f"eval_entrypoint={eval_entrypoint}",
        f"total_timesteps={total_timesteps}",
        f"protein_metric_path={metric_path}",
        # SweepTool overrides (post-construction)
        f"sweep_name={sweep_name}",
        f"wandb.entity={wandb_entity}",
        f"wandb.project={wandb_project}",
        f"dispatcher_type={dispatcher}",
        f"max_trials={max_trials}",
        f"batch_size={batch_size}",
        f"max_parallel_jobs={max_parallel_jobs}",
    ]


def run_experiment(
    *,
    wandb_entity: str = "metta-research",
    wandb_project: str = "hardware-scaling",
    gpu_counts: list[int] | None = None,
    node_counts: list[int] | None = None,
    max_trials: int = 50,
    batch_size: int = 1,
    max_parallel_jobs: int = 4,
    recipe: str = "experiments.recipes.arena_basic_easy_shaped",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    total_timesteps: int = 1_000_000_000,
    metric_path: str = "evaluator/eval_arena/score",
    dispatcher: str = "skypilot",
    experiment_name: str = "ak.hardware_scaling",
    sequential: bool = False,
    delay: int = 60,
) -> int:
    """Programmatic mirror of main() to launch hardware-aware sweeps.

    Returns 0 on success. Mirrors main() behavior, including waiting for child
    processes (sequentially or in parallel with a final wait).
    """
    init_logging()

    g_list = gpu_counts if gpu_counts is not None else [1, 2, 4, 6, 8, 12]
    n_list = node_counts if node_counts is not None else [1, 2, 4]
    pairs = _pairs(g_list, n_list)
    logger.info("Launching %d sweeps: %s", len(pairs), pairs)

    processes: list[subprocess.Popen] = []

    for gpus, nodes in pairs:
        suffix = f"g{gpus}n{nodes}"
        sweep_name = f"{experiment_name}.{suffix}"

        cmd = _build_cmd(
            gpus=gpus,
            nodes=nodes,
            sweep_name=sweep_name,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            dispatcher=dispatcher,
            max_trials=max_trials,
            batch_size=batch_size,
            max_parallel_jobs=max_parallel_jobs,
            recipe=recipe,
            train_entrypoint=train_entrypoint,
            eval_entrypoint=eval_entrypoint,
            total_timesteps=total_timesteps,
            metric_path=metric_path,
        )

        logger.info("Dispatching %s: %s", sweep_name, " ".join(cmd))
        proc = subprocess.Popen(cmd)

        if sequential:
            ret = proc.wait()
            logger.info("Sweep %s exited with code %s", sweep_name, ret)
            if ret != 0:
                logger.warning("Sweep %s failed (exit=%s)", sweep_name, ret)
        else:
            processes.append(proc)
            if delay > 0:
                time.sleep(delay)

    if not sequential:
        logger.info(
            "All sweeps launched; waiting for child processes to finish (Ctrl+C to exit)."
        )
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted; leaving child processes running in background.")

    logger.info("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch hardware-aware sweeps via CLI subprocesses"
    )

    # WandB settings
    parser.add_argument("--wandb-entity", type=str, required=True, help="WandB entity")
    parser.add_argument(
        "--wandb-project", type=str, default="hardware-scaling", help="WandB project"
    )

    # Hardware configurations
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        help="GPU counts to test",
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Node counts to test",
    )

    # Sweep knobs (passed to SweepTool)
    parser.add_argument(
        "--max-trials", type=int, default=50, help="Max trials per sweep"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Suggestions per batch"
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=4,
        help="Max parallel jobs in controller",
    )

    # Recipe & entrypoints
    parser.add_argument(
        "--recipe",
        type=str,
        default="experiments.recipes.arena_basic_easy_shaped",
        help="Recipe module to use",
    )
    parser.add_argument(
        "--train-entrypoint",
        type=str,
        default="train",
        help="Train entrypoint in recipe",
    )
    parser.add_argument(
        "--eval-entrypoint",
        type=str,
        default="evaluate",
        help="Eval entrypoint in recipe",
    )

    # Training & metric
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000_000,
        help="Total timesteps per trial",
    )
    parser.add_argument(
        "--metric-path",
        type=str,
        default="evaluator/eval_sweep/score",
        help="Metric path to optimize (summary key)",
    )

    # Dispatcher
    parser.add_argument(
        "--dispatcher",
        choices=["local", "skypilot"],
        default="local",
        help="Dispatcher type",
    )

    # Experiment naming and sequencing
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hardware_scaling",
        help="Base sweep name prefix",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run sequentially instead of parallel"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Delay between launches (seconds) if parallel",
    )

    args = parser.parse_args()

    return run_experiment(
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        gpu_counts=args.gpu_counts,
        node_counts=args.node_counts,
        max_trials=args.max_trials,
        batch_size=args.batch_size,
        max_parallel_jobs=args.max_parallel_jobs,
        recipe=args.recipe,
        train_entrypoint=args.train_entrypoint,
        eval_entrypoint=args.eval_entrypoint,
        total_timesteps=args.total_timesteps,
        metric_path=args.metric_path,
        dispatcher=args.dispatcher,
        experiment_name=args.experiment_name,
        sequential=args.sequential,
        delay=args.delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
