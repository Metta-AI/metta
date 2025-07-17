#!/usr/bin/env python3
"""
GPU-optimized script to run historical navigation evaluations on checkpoints from a training run.
This version uses parallel processing and efficient GPU memory management for faster evaluation.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metta.common.wandb.log_wandb import log_wandb


def find_checkpoint_artifacts(
    run_name: str, project: str = "metta", entity: str = "metta-research"
) -> List[Tuple[str, int, int]]:
    """
    Find all checkpoint artifacts associated with a wandb run.

    Returns:
        List of (artifact_name, epoch, agent_step) tuples
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_name}")

        # Get all artifacts created by this run
        artifacts = []
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                try:
                    # Extract metadata from the artifact
                    epoch = artifact.metadata.get("epoch", 0)
                    agent_step = artifact.metadata.get("agent_step", 0)
                    artifacts.append((artifact.name, epoch, agent_step))
                except Exception as e:
                    print(f"⚠️  Error reading artifact {artifact.name}: {e}")
                    continue

        # Sort by epoch
        artifacts.sort(key=lambda x: x[1])
        return artifacts

    except Exception as e:
        print(f"❌ Error accessing wandb run {run_name}: {e}")
        return []


def run_navigation_evaluation_gpu(
    artifact_name: str,
    run_name: str,
    epoch: int,
    device: str = "cuda",
    timeout: int = 3600,
    sim_type: str = "navigation",
) -> Optional[Dict[str, Any]]:
    """
    Run evaluation on a specific checkpoint using GPU.

    Returns:
        Dictionary of evaluation results, or None if failed
    """
    print(f"  🚀 Running {sim_type} evaluation on checkpoint: epoch {epoch} (GPU: {device})")

    # Create a unique eval run name
    eval_run_name = f"{run_name}_historical_eval_epoch_{epoch}"

    # Use the wandb artifact URI directly (format: wandb://entity/project/type/name:version)
    artifact_uri = f"wandb://metta-research/metta/model/{artifact_name}"

    # Use appropriate database URI based on sim_type
    if sim_type == "nav_sequence":
        db_uri = "wandb://stats/nav_sequence_db"
    elif sim_type == "memory":
        db_uri = "wandb://stats/memory_db"
    elif sim_type == "object_use":
        db_uri = "wandb://stats/objectuse_db"
    else:
        db_uri = "wandb://stats/navigation_db"

    cmd = [
        "./tools/sim.py",
        f"sim={sim_type}",
        f"run={eval_run_name}",
        f"policy_uri={artifact_uri}",
        f"sim_job.stats_db_uri={db_uri}",
        f"device={device}",
    ]

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"  ✅ Evaluation completed for epoch {epoch} in {duration:.1f}s")
            return {
                "success": True,
                "eval_run_name": eval_run_name,
                "duration": duration,
                "epoch": epoch,
                "artifact_name": artifact_name,
            }
        else:
            print(f"  ❌ Evaluation failed for epoch {epoch} after {duration:.1f}s")
            print(f"     Error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"  ⏰ Evaluation timed out for epoch {epoch} after {duration:.1f}s")
        return None
    except Exception as e:
        duration = time.time() - start_time
        print(f"  ❌ Error running evaluation for epoch {epoch}: {e}")
        return None


def setup_wandb_resumption(run_name: str, project: str = "metta", entity: str = "metta-research"):
    """
    Resume the original wandb run to log historical evaluations.

    Returns:
        wandb run object if successful, None otherwise
    """
    try:
        # Set up environment for wandb resumption
        os.environ["METTA_RUN_ID"] = run_name
        os.environ["WANDB_PROJECT"] = project
        os.environ["WANDB_ENTITY"] = entity

        # Resume the run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_name,
            resume="allow",
            reinit=True,
        )

        print(f"✅ Successfully resumed wandb run: {run_name}")
        return run

    except Exception as e:
        print(f"❌ Failed to resume wandb run: {e}")
        return None


def log_evaluation_results(run_name: str, epoch: int, agent_step: int, eval_results: Dict[str, Any]):
    """
    Log evaluation results back to the original wandb run.
    """
    # Log completion marker
    success = log_wandb(key="eval/historical_navigation_completed", value=1, step=agent_step, also_summary=False)

    # Log duration if available
    if "duration" in eval_results:
        log_wandb(
            key="eval/historical_navigation_duration_s",
            value=eval_results["duration"],
            step=agent_step,
            also_summary=False,
        )

    if success:
        print(f"  ✅ Logged historical evaluation at step {agent_step}")
    else:
        print("  ❌ Failed to log historical evaluation")


def get_available_gpus() -> List[str]:
    """Get list of available GPU devices."""
    if not torch.cuda.is_available():
        return []

    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def run_evaluations_parallel(
    eval_artifacts: List[Tuple[str, int, int]],
    run_name: str,
    devices: List[str],
    max_workers: int = None,
    timeout: int = 3600,
    sim_type: str = "navigation",
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run evaluations in parallel across multiple GPUs.

    Returns:
        Tuple of (successful_count, failed_count, results)
    """
    if max_workers is None:
        max_workers = len(devices)

    print(f"🚀 Running {len(eval_artifacts)} evaluations in parallel")
    print(f"   Devices: {devices}")
    print(f"   Max workers: {max_workers}")
    print(f"   Timeout: {timeout}s per evaluation")

    successful = 0
    failed = 0
    results = []
    device_cycle = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_info = {}
        for i, (artifact_name, epoch, agent_step) in enumerate(eval_artifacts):
            device = devices[device_cycle % len(devices)]
            device_cycle += 1

            future = executor.submit(
                run_navigation_evaluation_gpu, artifact_name, run_name, epoch, device, timeout, sim_type
            )
            future_to_info[future] = (artifact_name, epoch, agent_step, device, i + 1)

        # Process completed jobs
        for future in as_completed(future_to_info):
            artifact_name, epoch, agent_step, device, job_num = future_to_info[future]

            try:
                result = future.result()
                if result:
                    successful += 1
                    results.append(result)
                    print(f"[{job_num}/{len(eval_artifacts)}] ✅ Epoch {epoch} completed on {device}")
                else:
                    failed += 1
                    print(f"[{job_num}/{len(eval_artifacts)}] ❌ Epoch {epoch} failed on {device}")
            except Exception as e:
                failed += 1
                print(f"[{job_num}/{len(eval_artifacts)}] ❌ Epoch {epoch} exception on {device}: {e}")

    return successful, failed, results


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized historical navigation evaluations")
    parser.add_argument("run_name", help="Name of the training run")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--project", default="metta", help="Wandb project name")
    parser.add_argument("--entity", default="metta-research", help="Wandb entity name")
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be done")
    parser.add_argument("--max_workers", type=int, help="Maximum parallel workers (default: number of GPUs)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per evaluation in seconds")
    parser.add_argument(
        "--device", help="Specific device to use (e.g. cuda:0). If not specified, uses all available GPUs"
    )
    parser.add_argument("--batch_size", type=int, default=0, help="Process checkpoints in batches (0 = all at once)")
    parser.add_argument("--sim", default="navigation", help="Simulation type (navigation, nav_sequence, memory, etc.)")

    args = parser.parse_args()

    # Check GPU availability
    if args.device:
        devices = [args.device]
    else:
        devices = get_available_gpus()
        if not devices:
            print("⚠️  No GPUs available, falling back to CPU")
            devices = ["cpu"]

    print(f"🖥️  Available devices: {devices}")

    print(f"🔍 Finding checkpoint artifacts for run: {args.run_name}")
    artifacts = find_checkpoint_artifacts(args.run_name, args.project, args.entity)

    if not artifacts:
        print("❌ No checkpoint artifacts found")
        return 1

    print(f"📁 Found {len(artifacts)} checkpoint artifacts")

    # Filter artifacts by evaluation interval
    eval_artifacts = []
    for artifact_name, epoch, agent_step in artifacts:
        if epoch % args.eval_interval == 0:
            eval_artifacts.append((artifact_name, epoch, agent_step))

    print(
        f"📊 Will evaluate {len(eval_artifacts)} checkpoints (every {args.eval_interval} epochs) using {args.sim} eval"
    )

    if args.dry_run:
        print("\n🔍 DRY RUN - Would evaluate these checkpoints:")
        for artifact_name, epoch, agent_step in eval_artifacts:
            print(f"  Epoch {epoch:4d} (step {agent_step:8d}): {artifact_name}")
        return 0

    # Resume wandb run
    print(f"\n📊 Resuming wandb run: {args.run_name}")
    wandb_run = setup_wandb_resumption(args.run_name, args.project, args.entity)

    if not wandb_run:
        print("❌ Failed to resume wandb run")
        return 1

    # Run evaluations
    start_time = time.time()

    if args.batch_size > 0:
        # Process in batches
        print(f"📦 Processing in batches of {args.batch_size}")
        total_successful = 0
        total_failed = 0
        all_results = []

        for i in range(0, len(eval_artifacts), args.batch_size):
            batch = eval_artifacts[i : i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(eval_artifacts) + args.batch_size - 1) // args.batch_size

            print(f"\n🎯 Processing batch {batch_num}/{total_batches} ({len(batch)} checkpoints)")

            successful, failed, results = run_evaluations_parallel(
                batch, args.run_name, devices, args.max_workers, args.timeout, args.sim
            )

            # Log results for this batch
            for result in results:
                if result:
                    epoch = result["epoch"]
                    agent_step = next(step for _, e, step in batch if e == epoch)
                    log_evaluation_results(args.run_name, epoch, agent_step, result)

            total_successful += successful
            total_failed += failed
            all_results.extend(results)

            print(f"📊 Batch {batch_num} complete: {successful} successful, {failed} failed")

        successful = total_successful
        failed = total_failed
        results = all_results
    else:
        # Process all at once
        successful, failed, results = run_evaluations_parallel(
            eval_artifacts, args.run_name, devices, args.max_workers, args.timeout, args.sim
        )

        # Log all results
        for result in results:
            if result:
                epoch = result["epoch"]
                agent_step = next(step for _, e, step in eval_artifacts if e == epoch)
                log_evaluation_results(args.run_name, epoch, agent_step, result)

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print("🎯 Historical Evaluation Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(eval_artifacts)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📈 Average time per eval: {total_time / len(eval_artifacts):.1f}s")

    if successful > 0:
        avg_duration = sum(r["duration"] for r in results if r and "duration" in r) / successful
        print(f"⚡ Average evaluation duration: {avg_duration:.1f}s")

    if wandb_run:
        # Log summary statistics
        wandb_run.summary["historical_eval/checkpoints_evaluated"] = successful
        wandb_run.summary["historical_eval/checkpoints_failed"] = failed
        wandb_run.summary["historical_eval/eval_interval"] = args.eval_interval
        wandb_run.summary["historical_eval/total_duration"] = total_time
        wandb_run.summary["historical_eval/devices_used"] = len(devices)

        print(f"📊 Wandb run remains open: https://wandb.ai/{args.entity}/{args.project}/runs/{args.run_name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
