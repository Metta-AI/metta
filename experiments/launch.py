"""Core training launch functionality."""

import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from metta.common.util.fs import get_repo_root


def launch_training_run(
    run_name: str,
    curriculum: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    no_spot: bool = False,
    additional_args: Optional[List[str]] = None,
    wandb_tags: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Launch a single training run using skypilot.

    This is the core function used by both notebooks and Experiment classes.

    Args:
        run_name: Name for the training run
        curriculum: Path to curriculum config
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes
        no_spot: Whether to disable spot instances
        additional_args: Additional command line arguments
        wandb_tags: Tags for wandb
        dry_run: If True, print command but don't execute

    Returns:
        Dictionary containing:
            - job_id: Sky job ID
            - run_name: Wandb run name
            - success: Whether launch succeeded
            - command: Full command executed
            - output: Command output lines
    """
    # Build command
    cmd = [
        "./devops/skypilot/launch.py",
        "train",
        f"run={run_name}",
        f"--gpus={num_gpus}",
        f"--nodes={num_nodes}",
    ]

    if no_spot:
        cmd.append("--no-spot")

    cmd.append(f"trainer.curriculum={curriculum}")

    if wandb_tags:
        cmd.append(f"+wandb.tags={json.dumps(wandb_tags)}")

    if additional_args:
        cmd.extend(additional_args)

    result = {
        "job_id": None,
        "run_name": run_name,
        "success": False,
        "command": " ".join(cmd),
        "output": [],
        "timestamp": datetime.now().isoformat(),
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {result['command']}")
        result["success"] = True
        result["job_id"] = "dry-run-job-id"
        return result

    print(f"Launching training job: {run_name}")
    print(f"Command: {result['command']}")
    print("=" * 50)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=get_repo_root(),
        )

        for line in process.stdout or []:
            result["output"].append(line.strip())
            print(line, end="")

            # Extract job ID from output
            if "Job ID:" in line or "sky-" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("sky-") and "-" in part[4:]:
                        result["job_id"] = part

        process.wait()
        result["success"] = process.returncode == 0

        if result["success"]:
            print("\n✓ Job launched successfully!")
            if result["job_id"]:
                print(f"Job ID: {result['job_id']}")
        else:
            print(f"\n✗ Launch failed with return code: {process.returncode}")

    except Exception as e:
        print(f"\n✗ Error launching job: {str(e)}")
        result["output"].append(f"Error: {str(e)}")

    return result
