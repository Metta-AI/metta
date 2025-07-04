#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
# ]
# ///
"""
Tiny helper that converts SkyPilot ``SKYPILOT_TASK_ID`` env var to
"queue latency [s]" and logs it to wandb.

Recognized formats  (prefix  ➜  SkyPilot version):

*  ``sky-YYYY-MM-DD-HH-MM-SS-ffffff_<cluster>_<n>``             ≤ v0.9  (default)
*  ``managed-sky-YYYY‑…_<cluster>_<n>``                         v0.7+  (managed/spot)
*  ``sky-managed-YYYY‑…_<cluster>_<n>``                         v0.6   (early alpha)

If the env var is absent or malformed, :func:`queue_latency_s` returns ``None``.
"""

import datetime
import os
import re
import sys
from typing import Final

_EPOCH: Final = datetime.timezone.utc
_FMT: Final = "%Y-%m-%d-%H-%M-%S-%f"

# Regex captures the timestamp part, whatever the prefix is
# Updated to accept 6-9 digit microseconds (macOS can produce 9 digits)
_TS_RE: Final = re.compile(
    r"^(?:sky-|managed-sky-|sky-managed-)"
    r"(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6,9})_"
)


def _submission_ts(task_id: str) -> datetime.datetime | None:
    m = _TS_RE.match(task_id)
    if not m:
        return None
    # Truncate to 26 chars to ensure exactly 6 digits for microseconds
    ts_part = m.group("ts")[:26]  # YYYY-MM-DD-HH-MM-SS-ffffff (26 chars)
    try:
        return datetime.datetime.strptime(ts_part, _FMT).replace(tzinfo=_EPOCH)
    except ValueError:
        return None


def queue_latency_s() -> float | None:
    """Return SkyPilot queue latency in seconds, or ``None`` if not applicable."""
    task_id = os.environ.get("SKYPILOT_TASK_ID")
    if not task_id:
        return None
    submitted = _submission_ts(task_id)
    if submitted is None:
        return None
    return (datetime.datetime.now(_EPOCH) - submitted).total_seconds()


def main():
    """Log SkyPilot queue latency to stdout and optionally to wandb."""
    script_start_time = datetime.datetime.now(_EPOCH).isoformat()
    task_id = os.environ.get("SKYPILOT_TASK_ID", "unknown")
    run_id = os.environ.get("METTA_RUN_ID")

    # First, try to log to wandb that this script ran (regardless of latency calculation)
    if run_id:
        api_key = os.environ.get("WANDB_API_KEY")
        project = os.environ.get("WANDB_PROJECT", "metta")

        # If no API key but netrc exists, wandb will use that
        if api_key or os.path.exists(os.path.expanduser("~/.netrc")):
            try:
                import wandb

                # Only login if API key is explicitly provided
                if api_key:
                    wandb.login(key=api_key, relogin=True, anonymous="never")

                # Initialize wandb with the same run ID that the trainer will use
                # This creates a placeholder run that the trainer will resume
                run = wandb.init(
                    project=project,
                    name=run_id,
                    id=run_id,  # Use run_id as the unique wandb run ID
                    resume="allow",  # Allow resuming if it exists
                )

                # Always log that the script ran
                run.summary["skypilot/latency_script_ran"] = True
                run.summary["skypilot/latency_script_time"] = script_start_time

                # Now try to calculate and log the latency
                latency_sec = queue_latency_s()

                if latency_sec is not None:
                    print(f"SkyPilot queue latency: {latency_sec:.1f} s (task: {task_id})")

                    # Export for other scripts to use via environment
                    os.environ["SKYPILOT_QUEUE_LATENCY_S"] = str(latency_sec)

                    # Log the latency metrics
                    wandb.log(
                        {
                            "skypilot/queue_latency_s": latency_sec,
                            "skypilot/task_id": task_id,
                        },
                        step=0,
                    )

                    # Also add to summary for easy access
                    run.summary["skypilot/queue_latency_s"] = latency_sec
                    run.summary["skypilot/task_id"] = task_id
                    run.summary["skypilot/latency_calculated"] = True
                else:
                    print(f"SkyPilot queue latency: N/A (task_id: {task_id})")
                    run.summary["skypilot/latency_calculated"] = False
                    run.summary["skypilot/task_id"] = task_id
                    run.summary["skypilot/latency_error"] = "Could not parse task ID"

                # Don't call wandb.finish() - let the trainer resume this run
                print(f"✅ Logged to wandb run: {run_id}")
                entity = f"{os.environ.get('WANDB_ENTITY', wandb.api.default_entity)}"
                print(f"   View at: https://wandb.ai/{entity}/{project}/runs/{run_id}")

            except Exception as e:
                print(f"⚠️  Failed to log to wandb: {e}", file=sys.stderr)
        else:
            print("ℹ️  Skipping wandb logging (no API key or .netrc found)")
    else:
        print("ℹ️  Skipping wandb logging (METTA_RUN_ID not set)")

        # Still try to calculate latency for stdout
        latency_sec = queue_latency_s()
        if latency_sec is not None:
            print(f"SkyPilot queue latency: {latency_sec:.1f} s (task: {task_id})")
            os.environ["SKYPILOT_QUEUE_LATENCY_S"] = str(latency_sec)
        else:
            print("SkyPilot queue latency: N/A")

    return 0


if __name__ == "__main__":
    sys.exit(main())
