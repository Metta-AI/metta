#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
# ]
# ///
"""
Tiny helper that converts SkyPilot ``SKYPILOT_TASK_ID`` env var to
"queue latency [s]".

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
    task_id = os.environ.get("SKYPILOT_TASK_ID", "unknown")
    latency_sec = queue_latency_s()

    if latency_sec is not None:
        print(f"SkyPilot queue latency: {latency_sec:.1f} s (task: {task_id})")

        # Export for other scripts to use
        os.environ["SKYPILOT_QUEUE_LATENCY_S"] = str(latency_sec)

        # Also write to a file that persists across shell invocations
        latency_file = os.path.expanduser("~/.metta/skypilot_queue_latency")
        os.makedirs(os.path.dirname(latency_file), exist_ok=True)
        with open(latency_file, "w") as f:
            f.write(str(latency_sec))

        # Log to wandb if configured
        # Use METTA_RUN_ID as both the run name AND the run ID
        run_id = os.environ.get("METTA_RUN_ID")
        api_key = os.environ.get("WANDB_API_KEY")
        project = os.environ.get("WANDB_PROJECT", "metta")

        # If no API key but netrc exists, wandb will use that
        if run_id and (api_key or os.path.exists(os.path.expanduser("~/.netrc"))):
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
                    reinit=True,
                )

                # Log the latency metric
                run.summary["skypilot_queue_latency_s"] = latency_sec
                run.log({"skypilot_queue_latency_s": latency_sec}, step=0)

                # Don't call wandb.finish() - let the trainer resume this run
                print(f"Logged queue latency to wandb run: {run_id} (run will be resumed by trainer)")
            except Exception as e:
                print(f"Failed to log to wandb: {e}", file=sys.stderr)
    else:
        print("SkyPilot queue latency: N/A (not a SkyPilot job)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
