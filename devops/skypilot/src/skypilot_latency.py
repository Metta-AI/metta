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
import sys
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


def main() -> int:
    """Calculate and output SkyPilot queue latency."""
    try:
        latency_sec = calculate_queue_latency()

        # Output latency to stdout
        print(latency_sec)

        # Log info to stderr
        task_id = os.environ.get("SKYPILOT_TASK_ID", "unknown")
        print(f"SkyPilot queue latency: {latency_sec:.1f}s (task: {task_id})", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
