"""Recipe wrapper to run batch_fixed_maps.py on SkyPilot.

This recipe runs the batch training script that submits multiple training jobs.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from metta.cogworks.curriculum import CurriculumConfig, SingleTaskGenerator
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

# Import the batch script functions directly
# Since experiments might not be importable, we'll inline the logic
REPO_ROOT = Path(__file__).resolve().parents[2]
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
CHECKPOINT_EPOCH_INTERVAL = 25  # Save checkpoint every 25 epochs

# Activation extraction layers (to be implemented via hooks)
ACTIVATION_LAYERS = [1, 3, 5, 7]


def _build_skypilot_command(
    run_id: str,
    variants: Sequence[str],
    num_cogs: int,
    seed: int,
    total_timesteps: int,
    checkpoint_interval: int,
    extract_activations: bool,
    gpus: int,
    nodes: Optional[int],
    no_spot: bool,
    skip_git_check: bool,
    heartbeat_timeout: int,
    max_runtime_hours: Optional[float],
) -> list[str]:
    """Build SkyPilot launch command."""
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
    if extract_activations:
        activation_layers_str = json.dumps(ACTIVATION_LAYERS)
        command.append("+activation_extraction.enabled=true")
        command.append(f"+activation_extraction.layers={activation_layers_str}")

    return command


def _run_batch_submission() -> int:
    """Run batch submission for all conditions."""
    num_cogs = int(os.environ.get("BATCH_NUM_COGS", "4"))
    extract_activations = os.environ.get("BATCH_EXTRACT_ACTIVATIONS", "true").lower() == "true"
    seed = int(os.environ.get("BATCH_SEED", DEFAULT_SEED))
    seeds = [seed]

    gpus = int(os.environ.get("BATCH_GPUS", "8"))
    nodes_env = os.environ.get("BATCH_NODES")
    nodes = int(nodes_env) if nodes_env else None
    no_spot = os.environ.get("BATCH_NO_SPOT", "false").lower() == "true"
    skip_git_check = os.environ.get("BATCH_SKIP_GIT_CHECK", "true").lower() == "true"
    heartbeat_timeout = int(os.environ.get("BATCH_HEARTBEAT_TIMEOUT", "3600"))
    max_runtime_hours_env = os.environ.get("BATCH_MAX_RUNTIME_HOURS")
    max_runtime_hours = float(max_runtime_hours_env) if max_runtime_hours_env else None

    total_runs = len(RUN_DEFINITIONS) * len(seeds)
    current_run = 0

    print(f"[BATCH] Starting batch training: {len(RUN_DEFINITIONS)} conditions × 1 seed = {total_runs} total runs")
    print("[BATCH] Configuration:")
    print(f"  - GPUs: {gpus}")
    if nodes:
        print(f"  - Nodes: {nodes} ({gpus // nodes} GPUs per node)")
    print(f"  - Spot instances: {not no_spot}")
    print(f"  - Total timesteps per run: {TOTAL_TIMESTEPS:,}")
    print(f"  - Seed: {seed}")
    print()

    for condition_name, variants in RUN_DEFINITIONS:
        for seed_value in seeds:
            current_run += 1
            run_id = f"{condition_name}_seed{seed_value}"

            command = _build_skypilot_command(
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

            print(f"[BATCH] [{current_run}/{total_runs}] Submitting run {run_id}")
            print(f"  Condition: {condition_name}")
            print(f"  Variants: {list(variants) if variants else 'none'}")
            print(f"  Seed: {seed_value}")
            print(flush=True)

            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                print(f"[BATCH] Run {run_id} failed (exit code {completed.returncode})", flush=True)
                return completed.returncode

            print(f"[BATCH] Run {run_id} submitted successfully", flush=True)
            print()

    print(f"[BATCH] All {total_runs} runs submitted successfully")
    print("[BATCH] Monitor at: https://skypilot-api.softmax-research.net/")
    return 0


def train(
    batch_use_skypilot: Optional[bool] = None,
    batch_gpus: Optional[int] = None,
    batch_nodes: Optional[int] = None,
    batch_no_spot: Optional[bool] = None,
    batch_seed: Optional[int] = None,
    batch_num_cogs: Optional[int] = None,
    batch_extract_activations: Optional[bool] = None,
    batch_heartbeat_timeout: Optional[int] = None,
    batch_max_runtime_hours: Optional[float] = None,
) -> TrainTool:
    """Tool maker that runs the batch script to submit training jobs.

    Args:
        batch_use_skypilot: Enable SkyPilot mode (default: from BATCH_USE_SKYPILOT env var)
        batch_gpus: Number of GPUs per job (default: from BATCH_GPUS env var, default 8)
        batch_nodes: Number of nodes (default: from BATCH_NODES env var)
        batch_no_spot: Use on-demand instances (default: from BATCH_NO_SPOT env var)
        batch_seed: Random seed (default: from BATCH_SEED env var, default 42)
        batch_num_cogs: Number of cogs (default: from BATCH_NUM_COGS env var, default 4)
        batch_extract_activations: Enable activation extraction (default: from BATCH_EXTRACT_ACTIVATIONS env var)
        batch_heartbeat_timeout: Heartbeat timeout in seconds (default: from BATCH_HEARTBEAT_TIMEOUT env var)
        batch_max_runtime_hours: Max runtime in hours (default: from BATCH_MAX_RUNTIME_HOURS env var)
    """
    # Set environment variables from function arguments (if provided)
    # This allows CLI args to override env vars
    if batch_use_skypilot is not None:
        os.environ["BATCH_USE_SKYPILOT"] = "true" if batch_use_skypilot else "false"
    if batch_gpus is not None:
        os.environ["BATCH_GPUS"] = str(batch_gpus)
    if batch_nodes is not None:
        os.environ["BATCH_NODES"] = str(batch_nodes)
    if batch_no_spot is not None:
        os.environ["BATCH_NO_SPOT"] = "true" if batch_no_spot else "false"
    if batch_seed is not None:
        os.environ["BATCH_SEED"] = str(batch_seed)
    if batch_num_cogs is not None:
        os.environ["BATCH_NUM_COGS"] = str(batch_num_cogs)
    if batch_extract_activations is not None:
        os.environ["BATCH_EXTRACT_ACTIVATIONS"] = "true" if batch_extract_activations else "false"
    if batch_heartbeat_timeout is not None:
        os.environ["BATCH_HEARTBEAT_TIMEOUT"] = str(batch_heartbeat_timeout)
    if batch_max_runtime_hours is not None:
        os.environ["BATCH_MAX_RUNTIME_HOURS"] = str(batch_max_runtime_hours)

    class BatchSubmitterTool(TrainTool):
        def invoke(self, args: dict[str, str]) -> int | None:
            """Invoke batch submission."""
            return _run_batch_submission()

    # Provide minimal valid configs (not actually used - we're just submitting jobs)
    # These are required by TrainTool but won't be used since we override invoke()
    minimal_curriculum = CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=MettaGridConfig()),
    )
    return BatchSubmitterTool(
        training_env=TrainingEnvironmentConfig(curriculum=minimal_curriculum),
        evaluator=EvaluatorConfig(),
    )
