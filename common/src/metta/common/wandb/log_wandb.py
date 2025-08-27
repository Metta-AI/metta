#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
# ]
# ///
"""
General purpose utility to log values to wandb.
Can be used standalone or imported by other scripts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from metta.common.util.constants import METTA_WANDB_PROJECT


def ensure_wandb_run():
    """
    Ensure a wandb run exists, creating/resuming if needed.

    Returns:
        wandb.Run object

    Raises:
        RuntimeError: If no credentials or run ID available
    """
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("wandb not installed") from e

    # Check if run already exists
    if wandb.run is not None:
        return wandb.run

    # Need to create/resume a run
    run_id = os.environ.get("METTA_RUN_ID")
    if not run_id:
        raise RuntimeError("No active wandb run and METTA_RUN_ID not set")

    # Check credentials
    api_key = os.environ.get("WANDB_API_KEY")
    has_netrc = os.path.exists(os.path.expanduser("~/.netrc"))

    if not api_key and not has_netrc:
        raise RuntimeError("No wandb credentials (need WANDB_API_KEY or ~/.netrc)")

    project = os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT)

    # Login if API key provided
    if api_key:
        wandb.login(key=api_key, relogin=True, anonymous="never")

    # Create/resume run
    run = wandb.init(
        project=project,
        name=run_id,
        id=run_id,
        resume="allow",
        reinit=True,
    )

    entity = os.environ.get("WANDB_ENTITY", wandb.api.default_entity)
    print(f"✅ Wandb run: https://wandb.ai/{entity}/{project}/runs/{run_id}", file=sys.stderr)

    return run


def log_to_wandb(metrics: dict[str, Any], step: int = 0, also_summary: bool = True) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of key-value pairs to log
        step: The step to log at (default 0)
        also_summary: Whether to also add to wandb.summary (default True)

    Raises:
        RuntimeError: If logging fails
    """
    run = ensure_wandb_run()

    try:
        import wandb

        # Log all metrics
        wandb.log(metrics, step=step)

        # Also add to summary if requested
        if also_summary:
            for key, value in metrics.items():
                run.summary[key] = value

        print(f"✅ Logged {len(metrics)} metrics to wandb", file=sys.stderr)

    except Exception as e:
        raise RuntimeError(f"Failed to log to wandb: {e}") from e


def log_single_value(key: str, value: Any, step: int = 0, also_summary: bool = True) -> None:
    """
    Convenience function to log a single key-value pair.

    Args:
        key: Metric key
        value: Metric value
        step: Step to log at
        also_summary: Whether to add to summary
    """
    log_to_wandb({key: value}, step=step, also_summary=also_summary)


def log_debug_info() -> None:
    """Log various debug information about the environment."""
    debug_metrics = {
        "debug/timestamp": datetime.utcnow().isoformat(),
        "debug/skypilot_task_id": os.environ.get("SKYPILOT_TASK_ID", "not_set"),
        "debug/metta_run_id": os.environ.get("METTA_RUN_ID", "not_set"),
        "debug/wandb_project": os.environ.get("WANDB_PROJECT", "not_set"),
        "debug/hostname": os.environ.get("HOSTNAME", "unknown"),
        "debug/rank": os.environ.get("RANK", "not_set"),
        "debug/local_rank": os.environ.get("LOCAL_RANK", "not_set"),
    }

    print("Debug environment:", file=sys.stderr)
    for k, v in debug_metrics.items():
        print(f"  {k.split('/')[-1]}: {v}", file=sys.stderr)

    log_to_wandb(debug_metrics)


def parse_value(value_str: str) -> Any:
    """
    Try to parse a string value into appropriate type.

    Args:
        value_str: String representation of value

    Returns:
        Parsed value (int, float, bool, or original string)
    """
    # Try to parse as JSON first (handles dicts, lists, etc.)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass

    # Try numeric types
    try:
        value = float(value_str)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        pass

    # Check for boolean
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Return as string
    return value_str


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Log values to wandb",
        epilog="Examples:\n"
        "  %(prog)s my/metric 42\n"
        "  %(prog)s accuracy 0.95 --step 1000\n"
        "  echo 3.14 | %(prog)s accuracy/train\n"
        "  %(prog)s --debug\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("key", nargs="?", help="Metric key to log")
    parser.add_argument("value", nargs="?", help="Value to log (reads from stdin if not provided)")
    parser.add_argument("--step", type=int, default=0, help="Step to log at (default: 0)")
    parser.add_argument("--no-summary", action="store_true", help="Don't add to wandb summary")
    parser.add_argument("--debug", action="store_true", help="Log debug environment info")

    args = parser.parse_args()

    try:
        if args.debug:
            log_debug_info()
        else:
            if not args.key:
                parser.error("Key is required unless using --debug")

            # Get value from args or stdin
            if args.value is not None:
                value_str = args.value
            else:
                # Read from stdin
                value_str = sys.stdin.read().strip()
                if not value_str:
                    raise RuntimeError("No value provided (empty stdin)")

            # Parse value
            value = parse_value(value_str)

            # Log to wandb
            log_single_value(args.key, value, step=args.step, also_summary=not args.no_summary)

            # Echo value to stdout for chaining
            print(value)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
