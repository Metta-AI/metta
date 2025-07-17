#!/usr/bin/env python3
"""
Script to run historical navigation evaluations on checkpoints from a training run.
This allows retroactive evaluation of policies at different training stages and logs
the results back to the original wandb run as if they were done during training.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

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


def download_checkpoint_artifact(
    artifact_name: str, project: str = "metta", entity: str = "metta-research"
) -> Optional[str]:
    """
    Download a checkpoint artifact from wandb.

    Returns:
        Path to the downloaded checkpoint file, or None if failed
    """
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")

        # Download to a temporary directory
        temp_dir = tempfile.mkdtemp()
        artifact.download(root=temp_dir)

        # Find the model.pt file
        model_path = os.path.join(temp_dir, "model.pt")
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"❌ model.pt not found in artifact {artifact_name}")
            return None

    except Exception as e:
        print(f"❌ Error downloading artifact {artifact_name}: {e}")
        return None


def run_navigation_evaluation(artifact_name: str, run_name: str, epoch: int, device: str = "cpu") -> Optional[dict]:
    """
    Run navigation evaluation on a specific checkpoint.

    Returns:
        Dictionary of evaluation results, or None if failed
    """
    print(f"  Running navigation evaluation on checkpoint: epoch {epoch}")

    # Create a unique eval run name
    eval_run_name = f"{run_name}_historical_eval_epoch_{epoch}"

    # Use the wandb artifact URI directly (format: wandb://entity/project/type/name:version)
    artifact_uri = f"wandb://metta-research/metta/model/{artifact_name}"

    cmd = [
        "./tools/sim.py",
        "sim=navigation",
        f"run={eval_run_name}",
        f"policy_uri={artifact_uri}",
        "sim_job.stats_db_uri=wandb://stats/navigation_db",
        f"device={device}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.returncode == 0:
            print("  ✅ Evaluation completed")
            return {"success": True, "eval_run_name": eval_run_name}
        else:
            print(f"  ❌ Evaluation failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("  ⏰ Evaluation timed out")
        return None
    except Exception as e:
        print(f"  ❌ Error running evaluation: {e}")
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


def log_evaluation_results(run_name: str, epoch: int, agent_step: int, eval_results: dict):
    """
    Log evaluation results back to the original wandb run.

    This is a placeholder - in practice, you'd need to:
    1. Load the actual evaluation results from the stats database
    2. Compute the navigation scores
    3. Log them in the same format as training-time evaluations
    """
    # For now, just log that the evaluation was completed
    success = log_wandb(key="eval/historical_navigation_completed", value=1, step=agent_step, also_summary=False)

    if success:
        print(f"  ✅ Logged historical evaluation at step {agent_step}")
    else:
        print("  ❌ Failed to log historical evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run historical navigation evaluations on training checkpoints")
    parser.add_argument("run_name", help="Name of the training run (e.g., jacke.sky_comprehensive_20250716_145441)")
    parser.add_argument("--device", default="cpu", help="Device to use for evaluation")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N epochs")
    parser.add_argument("--project", default="metta", help="Wandb project name")
    parser.add_argument("--entity", default="metta-research", help="Wandb entity name")
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be done")

    args = parser.parse_args()

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

    print(f"📊 Will evaluate {len(eval_artifacts)} checkpoints (every {args.eval_interval} epochs)")

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
    print("\n🚀 Running historical evaluations...")
    successful = 0
    failed = 0

    for i, (artifact_name, epoch, agent_step) in enumerate(eval_artifacts, 1):
        print(f"\n[{i}/{len(eval_artifacts)}] Evaluating epoch {epoch} (step {agent_step})")

        # Run navigation evaluation using wandb artifact URI directly
        eval_results = run_navigation_evaluation(artifact_name, args.run_name, epoch, args.device)

        if eval_results:
            # Log results back to wandb
            log_evaluation_results(args.run_name, epoch, agent_step, eval_results)
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("Historical Evaluation Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(eval_artifacts)}")

    if wandb_run:
        # Log summary statistics
        wandb_run.summary["historical_eval/checkpoints_evaluated"] = successful
        wandb_run.summary["historical_eval/checkpoints_failed"] = failed
        wandb_run.summary["historical_eval/eval_interval"] = args.eval_interval

        # Don't finish - let the run remain open for potential future logging
        print(f"📊 Wandb run remains open: https://wandb.ai/{args.entity}/{args.project}/runs/{args.run_name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
