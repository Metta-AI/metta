#!/usr/bin/env python3

"""Batch training script for fixed_maps experiments.

Runs training for 8 conditions with 1 seed per batch (8 jobs total), with 50M timesteps each.
Includes activation extraction hooks for layers 1, 3, 5, and 7.

The seed can be configured via BATCH_SEED environment variable (default: 42).

Usage:
    # Local execution:
    python experiments/recipes/batch_fixed_maps.py

    # SkyPilot execution (8 GPUs, 1 node):
    BATCH_USE_SKYPILOT=true BATCH_GPUS=8 python experiments/recipes/batch_fixed_maps.py

    # SkyPilot execution (8 GPUs, 2 nodes = 4 GPUs per node):
    BATCH_USE_SKYPILOT=true BATCH_GPUS=8 BATCH_NODES=2 python experiments/recipes/batch_fixed_maps.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_LAUNCH_SCRIPT = REPO_ROOT / "devops" / "skypilot" / "launch.py"

# 8 conditions: 7 variant removal conditions + 1 full set with all variants
RUN_DEFINITIONS: Sequence[tuple[str, Sequence[str]]] = (
    ("my_fixed_maps_run_removed_lonely_heart", ("pack_rat", "heart_chorus")),
    ("my_fixed_maps_run_removed_pack_rat", ("lonely_heart", "heart_chorus")),
    ("my_fixed_maps_run_removed_heart_chorus", ("lonely_heart", "pack_rat")),
    ("my_fixed_maps_run_removed_pack_rat_heart_chorus", ("lonely_heart",)),
    ("my_fixed_maps_run_removed_lonely_heart_heart_chorus", ("pack_rat",)),
    ("my_fixed_maps_run_removed_lonely_heart_pack_rat", ("heart_chorus",)),
    ("my_fixed_maps_run_removed_all_variants", ()),
    ("my_fixed_maps_run_all_variants", ("lonely_heart", "heart_chorus", "pack_rat")),
)

# Default seed (configurable via BATCH_SEED environment variable)
DEFAULT_SEED = 42

# Training configuration
TOTAL_TIMESTEPS = 50_000_000  # 50M timesteps
CHECKPOINT_EPOCH_INTERVAL = 25  # Save checkpoint every 25 epochs (reasonable for 50M steps)

# Activation extraction layers (to be implemented via hooks)
ACTIVATION_LAYERS = [1, 3, 5, 7]


def build_local_command(
    run_id: str,
    variants: Sequence[str],
    num_cogs: int,
    seed: int,
    total_timesteps: int,
    checkpoint_interval: int,
    extract_activations: bool = True,
) -> list[str]:
    """Build local training command with all required parameters."""
    command = [
        "uv",
        "run",
        "./tools/run.py",
        "recipes.prod.cvc.fixed_maps.train",
        f"run={run_id}",
        f"num_cogs={num_cogs}",
        f"variants={json.dumps(list(variants))}",
        f"system.seed={seed}",
        f"trainer.total_timesteps={total_timesteps}",
        f"checkpointer.epoch_interval={checkpoint_interval}",
    ]

    # Add activation extraction configuration if enabled
    # Note: This assumes the training infrastructure will support these parameters
    # The actual hook implementation will be added separately
    if extract_activations:
        activation_layers_str = json.dumps(ACTIVATION_LAYERS)
        command.append("+activation_extraction.enabled=true")
        command.append(f"+activation_extraction.layers={activation_layers_str}")

    return command


def build_skypilot_command(
    run_id: str,
    variants: Sequence[str],
    num_cogs: int,
    seed: int,
    total_timesteps: int,
    checkpoint_interval: int,
    extract_activations: bool = True,
    gpus: int = 8,
    nodes: Optional[int] = None,
    no_spot: bool = False,
    skip_git_check: bool = True,
    heartbeat_timeout: int = 3600,
    max_runtime_hours: Optional[float] = None,
) -> list[str]:
    """Build SkyPilot launch command with all required parameters."""
    if not SKYPILOT_LAUNCH_SCRIPT.exists():
        raise FileNotFoundError(
            f"SkyPilot launch script not found at {SKYPILOT_LAUNCH_SCRIPT}. "
            "Make sure you're running from the repo root."
        )

    command = [
        str(SKYPILOT_LAUNCH_SCRIPT),
        "recipes.prod.cvc.fixed_maps.train",
        f"run={run_id}",
        f"num_cogs={num_cogs}",
        f"variants={json.dumps(list(variants))}",
        f"system.seed={seed}",
        f"trainer.total_timesteps={total_timesteps}",
        f"checkpointer.epoch_interval={checkpoint_interval}",
        f"--gpus={gpus}",
        f"--heartbeat-timeout={heartbeat_timeout}",
    ]

    if nodes is not None:
        command.append(f"--nodes={nodes}")

    if no_spot:
        command.append("--no-spot")

    if skip_git_check:
        command.append("--skip-git-check")

    if max_runtime_hours is not None:
        command.append(f"--max-runtime-hours={max_runtime_hours}")

    # Add activation extraction configuration if enabled
    if extract_activations:
        activation_layers_str = json.dumps(ACTIVATION_LAYERS)
        command.append("+activation_extraction.enabled=true")
        command.append(f"+activation_extraction.layers={activation_layers_str}")

    return command


def main() -> int:
    """Run batch training for all conditions with a single seed."""
    # Configuration from environment variables
    num_cogs = int(os.environ.get("BATCH_NUM_COGS", "4"))
    extra_args_env = os.environ.get("BATCH_EXTRA_ARGS", "")
    extra_args = [arg for arg in extra_args_env.split() if arg]
    extract_activations = os.environ.get("BATCH_EXTRACT_ACTIVATIONS", "true").lower() == "true"

    # Get seed from environment or use default (1 seed per batch)
    seed = int(os.environ.get("BATCH_SEED", DEFAULT_SEED))
    seeds = [seed]  # Single seed per batch

    # SkyPilot configuration
    use_skypilot = os.environ.get("BATCH_USE_SKYPILOT", "false").lower() == "true"
    gpus = int(os.environ.get("BATCH_GPUS", "8"))
    nodes_env = os.environ.get("BATCH_NODES")
    nodes = int(nodes_env) if nodes_env else None
    no_spot = os.environ.get("BATCH_NO_SPOT", "false").lower() == "true"
    skip_git_check = os.environ.get("BATCH_SKIP_GIT_CHECK", "true").lower() == "true"
    heartbeat_timeout = int(os.environ.get("BATCH_HEARTBEAT_TIMEOUT", "3600"))
    max_runtime_hours_env = os.environ.get("BATCH_MAX_RUNTIME_HOURS")
    max_runtime_hours = float(max_runtime_hours_env) if max_runtime_hours_env else None

    env = os.environ.copy()

    total_runs = len(RUN_DEFINITIONS) * len(seeds)
    current_run = 0

    conditions_count = len(RUN_DEFINITIONS)
    seeds_count = len(seeds)
    msg = (
        f"[BATCH] Starting batch training: {conditions_count} conditions × "
        f"{seeds_count} seed{'s' if seeds_count > 1 else ''} = {total_runs} total runs"
    )
    print(msg)
    print("[BATCH] Configuration:")
    print(f"  - Execution mode: {'SkyPilot' if use_skypilot else 'Local'}")
    if use_skypilot:
        print(f"  - GPUs: {gpus}")
        if nodes:
            print(f"  - Nodes: {nodes} ({gpus // nodes} GPUs per node)")
        print(f"  - Spot instances: {not no_spot}")
        print(f"  - Heartbeat timeout: {heartbeat_timeout}s")
        if max_runtime_hours:
            print(f"  - Max runtime: {max_runtime_hours} hours")
    print(f"  - Total timesteps per run: {TOTAL_TIMESTEPS:,}")
    print(f"  - Checkpoint interval: {CHECKPOINT_EPOCH_INTERVAL} epochs")
    print(f"  - Activation extraction: {extract_activations} (layers {ACTIVATION_LAYERS})")
    print(f"  - Seed: {seed}")
    print()

    for condition_name, variants in RUN_DEFINITIONS:
        for seed_value in seeds:
            current_run += 1
            # Include seed in run_id for uniqueness
            run_id = f"{condition_name}_seed{seed_value}"

            if use_skypilot:
                command = build_skypilot_command(
                    run_id=run_id,
                    variants=variants,
                    num_cogs=num_cogs,
                    seed=seed_value,
                    total_timesteps=TOTAL_TIMESTEPS,
                    checkpoint_interval=CHECKPOINT_EPOCH_INTERVAL,
                    extract_activations=extract_activations,
                    gpus=gpus,
                    nodes=nodes,
                    no_spot=no_spot,
                    skip_git_check=skip_git_check,
                    heartbeat_timeout=heartbeat_timeout,
                    max_runtime_hours=max_runtime_hours,
                )
            else:
                command = (
                    build_local_command(
                        run_id=run_id,
                        variants=variants,
                        num_cogs=num_cogs,
                        seed=seed_value,
                        total_timesteps=TOTAL_TIMESTEPS,
                        checkpoint_interval=CHECKPOINT_EPOCH_INTERVAL,
                        extract_activations=extract_activations,
                    )
                    + extra_args
                )

            print(f"[BATCH] [{current_run}/{total_runs}] Submitting run {run_id}")
            print(f"  Condition: {condition_name}")
            print(f"  Variants: {list(variants) if variants else 'none'}")
            print(f"  Seed: {seed_value}")
            print(f"  Command: {' '.join(command)}")
            print(flush=True)

            completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
            if completed.returncode != 0:
                print(f"[BATCH] Run {run_id} failed (exit code {completed.returncode})", flush=True)
                if use_skypilot:
                    print(
                        "[BATCH] Note: SkyPilot jobs are submitted asynchronously. "
                        "Check the SkyPilot dashboard for job status.",
                        flush=True,
                    )
                return completed.returncode

            if use_skypilot:
                print(f"[BATCH] Run {run_id} submitted to SkyPilot successfully", flush=True)
            else:
                print(f"[BATCH] Run {run_id} completed successfully", flush=True)
            print()

    if use_skypilot:
        print(f"[BATCH] All {total_runs} runs submitted to SkyPilot successfully", flush=True)
        print(
            "[BATCH] Monitor job status at: https://skypilot-api.softmax-research.net/",
            flush=True,
        )
    else:
        print(f"[BATCH] All {total_runs} runs completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
