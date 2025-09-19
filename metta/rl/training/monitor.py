"""Trainer component responsible for setting up system and memory monitoring."""

from __future__ import annotations

import logging
from typing import Optional

from metta.rl.training.component import TrainerComponent
from mettagrid.profiling.memory_monitor import MemoryMonitor
from mettagrid.profiling.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


def _setup_monitoring(policy, experience, timer) -> tuple[MemoryMonitor, SystemMonitor]:
    """Instantiate memory/system monitors for the trainer."""

    memory_monitor = MemoryMonitor()
    memory_monitor.add(experience, name="Experience", track_attributes=True)
    memory_monitor.add(policy, name="Policy", track_attributes=False)

    system_monitor = SystemMonitor(
        sampling_interval_sec=1.0,
        history_size=100,
        log_level=logger.getEffectiveLevel(),
        auto_start=True,
        external_timer=timer,
    )

    logger.info("Initialized memory and system monitoring")
    return memory_monitor, system_monitor


def _cleanup_monitoring(memory_monitor: MemoryMonitor | None, system_monitor: SystemMonitor | None) -> None:
    """Dispose of monitor resources when training ends."""

    if memory_monitor:
        memory_monitor.clear()
        logger.debug("Cleared memory monitor")

    if system_monitor:
        system_monitor.stop()
        logger.debug("Stopped system monitor")


class Monitor(TrainerComponent):
    """Manage memory and system monitors independently of stats reporting."""

    _master_only = True

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__()
        self._enabled = enabled
        self._memory_monitor: Optional[MemoryMonitor] = None
        self._system_monitor: Optional[SystemMonitor] = None

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._enabled:
            context.memory_monitor = None
            context.system_monitor = None
            return

        memory_monitor, system_monitor = _setup_monitoring(
            policy=context.policy,
            experience=context.experience,
            timer=context.stopwatch,
        )
        self._memory_monitor = memory_monitor
        self._system_monitor = system_monitor
        context.memory_monitor = memory_monitor
        context.system_monitor = system_monitor

    def on_training_complete(self) -> None:
        self._teardown()

    def on_failure(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if not self._enabled:
            return
        _cleanup_monitoring(self._memory_monitor, self._system_monitor)
        self.context.memory_monitor = None
        self.context.system_monitor = None
        self._memory_monitor = None
        self._system_monitor = None
