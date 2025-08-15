#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
# ]
# ///
"""
Simple utility to log debug values to wandb for testing.
Can be used standalone or imported into other scripts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from metta.common.util.constants import METTA_WANDB_PROJECT


def log_wandb(key: str, value: Any, step: int = 0, also_summary: bool = True) -> bool:
    """
    Log a key-value pair to wandb if a run exists.

    Args:
        key: The metric key name
        value: The value to log
        step: The step to log at (default 0)
        also_summary: Whether to also add to wandb.summary (default True)

    Returns:
        True if logged successfully, False otherwise
    """
    try:
        import wandb
    except ImportError:
        print(f"[log_wandb] wandb not installed, skipping: {key}={value}")
        return False

    # Get the current run (either existing or create new one)
    run = wandb.run

    # If no active run, try to create/resume one
    if run is None:
        # Try to resume the run based on METTA_RUN_ID
        run_id = os.environ.get("METTA_RUN_ID")
        project = os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT)

        if not run_id:
            print(f"[log_wandb] No active wandb run and no METTA_RUN_ID, skipping: {key}={value}")
            return False

        print(f"[log_wandb] Attempting to resume run: {run_id}")

        try:
            # Try to resume the existing run
            run = wandb.init(
                project=project,
                name=run_id,
                id=run_id,
                resume="allow",
                reinit=True,
            )
            print(f"[log_wandb] Successfully resumed wandb run: {run_id}")
        except Exception as e:
            print(f"[log_wandb] Failed to resume wandb run: {e}")
            return False

    # Now we should have a valid run object
    if run is None:
        print("[log_wandb] Error: run is still None after init attempt")
        return False

    try:
        # Log the metric
        wandb.log({key: value}, step=step)
        print(f"[log_wandb] Logged: {key}={value} at step={step}")

        # Also add to summary if requested
        if also_summary:
            run.summary[key] = value
            print(f"[log_wandb] Added to summary: {key}={value}")

        return True

    except Exception as e:
        print(f"[log_wandb] Error logging to wandb: {e}")
        return False


def log_debug_info():
    """Log various debug information to help diagnose issues."""
    debug_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "skypilot_task_id": os.environ.get("SKYPILOT_TASK_ID", "not_set"),
        "metta_run_id": os.environ.get("METTA_RUN_ID", "not_set"),
        "wandb_project": os.environ.get("WANDB_PROJECT", "not_set"),
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "rank": os.environ.get("RANK", "not_set"),
        "local_rank": os.environ.get("LOCAL_RANK", "not_set"),
    }

    print("[log_debug_info] Debug environment:")
    for k, v in debug_info.items():
        print(f"  {k}: {v}")

    # Try to log each piece of debug info
    for k, v in debug_info.items():
        log_wandb(f"debug/{k}", v)

    # Log a simple test value
    log_wandb("debug/test_value", 42)
    log_wandb("debug/test_float", 3.14159)

    # Try to read and log skypilot latency if available
    latency_file = os.path.expanduser("~/.metta/skypilot_latency.json")
    if os.path.exists(latency_file):
        try:
            with open(latency_file, "r") as f:
                latency_data = json.load(f)
            latency_sec = latency_data.get("latency_s")
            if latency_sec is not None:
                log_wandb("debug/skypilot_queue_latency_s", latency_sec)
                print(f"[log_debug_info] Found and logged latency: {latency_sec}s")
        except Exception as e:
            print(f"[log_debug_info] Error reading latency file: {e}")
    else:
        print(f"[log_debug_info] No latency file found at {latency_file}")


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description="Log debug values to wandb")
    parser.add_argument("key", nargs="?", default="debug/test", help="Key to log")
    parser.add_argument("value", nargs="?", default=42, help="Value to log")
    parser.add_argument("--debug", action="store_true", help="Log full debug info")

    args = parser.parse_args()

    if args.debug:
        log_debug_info()
    else:
        # Try to convert value to appropriate type
        try:
            value = float(args.value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = args.value

        success = log_wandb(args.key, value)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
