#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""
Calculate SkyPilot queue latency from SKYPILOT_TASK_ID env var.

Expected format: sky-YYYY-MM-DD-HH-MM-SS-ffffff_<cluster>_<n>
"""

import datetime
import os
import re
from typing import Final

_EPOCH: Final = datetime.timezone.utc
_FMT: Final = "%Y-%m-%d-%H-%M-%S-%f"

# Regex for SkyPilot task ID format
_TS_RE: Final = re.compile(r"^sky(?:-managed)?-(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6,9})_")


def parse_submission_timestamp(task_id: str) -> datetime.datetime:
    m = _TS_RE.match(task_id)
    if not m:
        raise ValueError(f"Invalid task ID format: {task_id}")

    # Truncate to 26 chars to ensure exactly 6 digits for microseconds
    ts_part = m.group("ts")[:26]  # YYYY-MM-DD-HH-MM-SS-ffffff (26 chars)
    try:
        return datetime.datetime.strptime(ts_part, _FMT).replace(tzinfo=_EPOCH)
    except ValueError as e:
        raise ValueError(f"Failed to parse timestamp from task ID: {task_id}") from e


def calculate_queue_latency() -> float:
    """Calculate SkyPilot queue latency in seconds."""
    task_id = os.environ.get("SKYPILOT_TASK_ID")
    if not task_id:
        raise RuntimeError("SKYPILOT_TASK_ID environment variable not set")

    submitted = parse_submission_timestamp(task_id)
    return (datetime.datetime.now(_EPOCH) - submitted).total_seconds()


if __name__ == "__main__":
    latency = calculate_queue_latency()

    # script_start_time = datetime.datetime.now(_EPOCH).isoformat()
    # task_id = os.environ.get("SKYPILOT_TASK_ID", "unknown")
    # run_id = os.environ.get("METTA_RUN_ID")

    # # First, try to log to wandb that this script ran (regardless of latency calculation)
    # if run_id:
    #     api_key = os.environ.get("WANDB_API_KEY")
    #     project = os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT)

    #     # If no API key but netrc exists, wandb will use that
    #     if api_key or os.path.exists(os.path.expanduser("~/.netrc")):
    #         try:
    #             import wandb

    #             # Only login if API key is explicitly provided
    #             if api_key:
    #                 wandb.login(key=api_key, relogin=True, anonymous="never")

    #             # Initialize wandb with the same run ID that the trainer will use
    #             # This creates a placeholder run that the trainer will resume
    #             run = wandb.init(
    #                 project=project,
    #                 name=run_id,
    #                 id=run_id,  # Use run_id as the unique wandb run ID
    #                 resume="allow",  # Allow resuming if it exists
    #             )

    #             # Always log that the script ran
    #             run.summary["skypilot/latency_script_ran"] = True
    #             run.summary["skypilot/latency_script_time"] = script_start_time

    #             # Now try to calculate and log the latency
    #             latency_sec = queue_latency_s()

    #             if latency_sec is not None:
    #                 print(f"SkyPilot queue latency: {latency_sec:.1f} s (task: {task_id})")

    #                 # Export for other scripts to use via environment
    #                 os.environ["SKYPILOT_QUEUE_LATENCY_S"] = str(latency_sec)

    #                 # Also add to summary for easy access
    #                 run.summary["skypilot/queue_latency_s"] = latency_sec
    #                 run.summary["skypilot/task_id"] = task_id
    #                 run.summary["skypilot/latency_calculated"] = True
    #             else:
    #                 print(f"SkyPilot queue latency: N/A (task_id: {task_id})")
    #                 run.summary["skypilot/latency_calculated"] = False
    #                 run.summary["skypilot/task_id"] = task_id
    #                 run.summary["skypilot/latency_error"] = "Could not parse task ID"

    #             # Don't call wandb.finish() - let the trainer resume this run
    #             print(f"✅ Logged to wandb run: {run_id}")
    #             entity = f"{os.environ.get('WANDB_ENTITY', wandb.api.default_entity)}"
    #             print(f"   View at: https://wandb.ai/{entity}/{project}/runs/{run_id}")

    #         except Exception as e:
    #             print(f"⚠️  Failed to log to wandb: {e}", file=sys.stderr)
    #     else:
    #         print("ℹ️  Skipping wandb logging (no API key or .netrc found)")
