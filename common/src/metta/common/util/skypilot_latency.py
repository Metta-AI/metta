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
_FMT:   Final = "%Y-%m-%d-%H-%M-%S-%f"

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

        # Log to wandb if configured
        # Use METTA_RUN_ID as the run name (set by SkyPilot launch script)
        run_name = os.environ.get("METTA_RUN_ID")
        api_key = os.environ.get("WANDB_API_KEY")

        # If no API key but netrc exists, wandb will use that
        if run_name and (api_key or os.path.exists(os.path.expanduser("~/.netrc"))):
            try:
                import wandb

                # Only login if API key is explicitly provided
                if api_key:
                    wandb.login(key=api_key, relogin=True, anonymous="never")

                # Initialize wandb with minimal config
                run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "metta"),
                    name=run_name,
                    resume="allow",
                    reinit=True,
                )
                run.summary["skypilot_queue_latency_s"] = latency_sec
                run.log({"skypilot_queue_latency_s": latency_sec}, step=0)
                wandb.finish()
                print(f"Logged queue latency to wandb run: {run_name}")
            except Exception as e:
                print(f"Failed to log to wandb: {e}", file=sys.stderr)
    else:
        print("SkyPilot queue latency: N/A (not a SkyPilot job)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
