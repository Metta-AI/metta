"""
Tiny helper that converts SkyPilot ``SKYPILOT_TASK_ID`` env var to
“queue latency [s]”.

Recognized formats  (prefix  ➜  SkyPilot version):

*  ``sky-YYYY-MM-DD-HH-MM-SS-ffffff_<cluster>_<n>``             ≤ v0.9  (default)
*  ``managed-sky-YYYY‑…_<cluster>_<n>``                         v0.7+  (managed/spot)
*  ``sky-managed-YYYY‑…_<cluster>_<n>``                         v0.6   (early alpha)

If the env var is absent or malformed, :func:`queue_latency_s` returns ``None``.
"""

import datetime
import os
import re
from typing import Final

_EPOCH: Final = datetime.timezone.utc
_FMT:   Final = "%Y-%m-%d-%H-%M-%S-%f"

# Regex captures the timestamp part, whatever the prefix is
_TS_RE: Final = re.compile(
    r"^(?:sky-|managed-sky-|sky-managed-)"
    r"(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6})_"
)

def _submission_ts(task_id: str) -> datetime.datetime | None:
    m = _TS_RE.match(task_id)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group("ts"), _FMT).replace(tzinfo=_EPOCH)
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
