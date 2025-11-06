#!/usr/bin/env -S uv run
"""Wrapper script that runs cogames training followed by evaluation.

This script:
1. Runs cogames train with --log-outputs
2. Checks for training errors
3. Finds the final checkpoint
4. Runs cogames eval on the checkpoint
5. Outputs evaluation results in parseable format

Usage:
    cogames_train_eval.py --mission MISSION --variant VAR1 --variant VAR2 \\
        --steps STEPS --checkpoints-dir DIR [--eval-episodes N]
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Find the most recent checkpoint file in the directory.

    Checkpoints are named like: cogames.cogs_vs_clips_model_NNNN.pt
    """
    if not checkpoints_dir.exists():
        return None

    checkpoints = list(checkpoints_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time, return most recent
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cogames training + evaluation")
    parser.add_argument("--mission", required=True, help="Mission name")
    parser.add_argument("--variant", action="append", default=[], help="Mission variants")
    parser.add_argument("--steps", type=int, required=True, help="Training steps")
    parser.add_argument("--checkpoints-dir", type=Path, required=True, help="Checkpoints directory")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--artifacts", required=True, help="JSON dict of artifact names to S3 URIs")
    parser.add_argument("--policy", default="lstm", help="Policy type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Parse artifacts JSON
    artifact_paths = json.loads(args.artifacts)

    # Ensure checkpoints directory exists
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Build training command
    train_cmd = [
        "uv",
        "run",
        "cogames",
        "train",
        "--mission",
        args.mission,
        "--steps",
        str(args.steps),
        "--checkpoints",
        str(args.checkpoints_dir),
        "--policy",
        args.policy,
        "--seed",
        str(args.seed),
        "--log-outputs",
    ]

    for variant in args.variant:
        train_cmd.extend(["--variant", variant])

    print(f"Training: {datetime.now(UTC)}", flush=True)
    print(f"Command: {' '.join(train_cmd)}", flush=True)

    # Run training
    train_result = subprocess.run(train_cmd, capture_output=False, text=True)

    if train_result.returncode != 0:
        print(f"Training failed with exit code {train_result.returncode}", file=sys.stderr, flush=True)
        return train_result.returncode

    print(f"Training complete: {datetime.now(UTC)}", flush=True)

    # Find the latest checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoints_dir)
    if not checkpoint_path:
        print(f"No checkpoint found in {args.checkpoints_dir}", file=sys.stderr, flush=True)
        return 1

    print(f"Found checkpoint: {checkpoint_path}", flush=True)

    # Build evaluation command
    policy_arg = f"{args.policy}:{checkpoint_path}"
    eval_cmd = [
        "uv",
        "run",
        "cogames",
        "eval",
        "--mission",
        args.mission,
        "--policy",
        policy_arg,
        "--episodes",
        str(args.eval_episodes),
    ]

    for variant in args.variant:
        eval_cmd.extend(["--variant", variant])

    print(f"Evaluation: {datetime.now(UTC)}", flush=True)
    print(f"Command: {' '.join(eval_cmd)}", flush=True)

    # Run evaluation and capture output
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

    if eval_result.returncode != 0:
        print(f"Evaluation failed with exit code {eval_result.returncode}", file=sys.stderr, flush=True)
        print(eval_result.stderr, file=sys.stderr, flush=True)
        return eval_result.returncode

    # Output eval results in single-line format for parsing (for log-based fallback)
    # The eval command with --format json outputs a JSON object
    print(f"EvalResults: {datetime.now(UTC)} {eval_result.stdout.strip()}", flush=True)

    # Upload eval results to S3 using paths from artifacts dict
    if "eval_results.json" in artifact_paths:
        try:
            s3_path = artifact_paths["eval_results.json"]

            # Write eval JSON to temp file
            eval_json_file = args.checkpoints_dir / "eval_results.json"
            eval_json_file.write_text(eval_result.stdout)

            # Upload to S3 using aws cli
            upload_cmd = ["aws", "s3", "cp", str(eval_json_file), s3_path]
            upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=30)

            if upload_result.returncode == 0:
                print(f"Uploaded eval results to {s3_path}", flush=True)
            else:
                print(
                    f"Warning: Failed to upload eval results to S3: {upload_result.stderr}", file=sys.stderr, flush=True
                )
        except Exception as e:
            print(f"Warning: Error uploading eval results to S3: {e}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
