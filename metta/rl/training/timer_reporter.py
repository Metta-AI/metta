"""Optional stopwatch dump for quick profiling.

Enable via METTA_TIMER_REPORT=1; logs per-timer elapsed seconds at training end.
"""

from __future__ import annotations

import json
import logging
import os

from metta.rl.training import ComponentContext, TrainerComponent

logger = logging.getLogger(__name__)


def _env_enabled() -> bool:
    return os.environ.get("METTA_TIMER_REPORT", "0") == "1"


class TimerReporter(TrainerComponent):
    def __init__(self) -> None:
        super().__init__(epoch_interval=0)
        self._master_only = True

    def register(self, context: ComponentContext) -> None:  # type: ignore[override]
        super().register(context)
        logger.info("TimerReporter enabled (METTA_TIMER_REPORT=1)")

    def on_training_complete(self) -> None:  # type: ignore[override]
        timer = self.context.stopwatch
        timings = timer.get_all_elapsed()
        sorted_timings = dict(sorted(timings.items(), key=lambda kv: kv[0]))
        logger.info("TimerReporter elapsed seconds per timer:\n%s", json.dumps(sorted_timings, indent=2))
