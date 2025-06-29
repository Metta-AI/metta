"""
Tiny helper that turns SkyPilot's SKYPILOT_TASK_ID env‑var into
a "queue latency in seconds" number.

Format of the var (set by SkyPilot on every worker process):

    sky-YYYY-MM-DD-HH-MM-SS-ffffff_<cluster>_<n>
"""

from __future__ import annotations

import datetime as _dt
import os as _os
from typing import Final

_EPOCH: Final = _dt.timezone.utc
_FMT: Final = "%Y-%m-%d-%H-%M-%S-%f"


def _submission_ts(task_id: str) -> _dt.datetime | None:
    if not task_id.startswith("sky-"):
        return None
    ts = task_id.split("_", 1)[0][4:]  # strip "sky-"
    try:
        return _dt.datetime.strptime(ts, _FMT).replace(tzinfo=_EPOCH)
    except ValueError:
        return None


def queue_latency_s() -> float | None:
    task_id = _os.environ.get("SKYPILOT_TASK_ID")
    if not task_id:
        return None
    submitted = _submission_ts(task_id)
    if submitted is None:
        return None
    return (_dt.datetime.now(_EPOCH) - submitted).total_seconds()
