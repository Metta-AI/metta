from __future__ import annotations

import logging
import os
import time


_TRUE_VALUES = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get("METTA_STARTUP_TIMING", "").strip().lower() in _TRUE_VALUES


def now() -> float:
    return time.perf_counter()


def log(logger: logging.Logger, label: str, start: float, end: float | None = None) -> None:
    if not enabled():
        return
    duration = (end if end is not None else time.perf_counter()) - start
    logger.info("startup/%s: %.3fs", label, duration)
