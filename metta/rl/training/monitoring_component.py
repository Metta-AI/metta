"""Trainer component responsible for setting up system and memory monitoring."""

from __future__ import annotations

from typing import Optional

from metta.core.monitoring import cleanup_monitoring, setup_monitoring
from metta.rl.training.component import TrainerComponent
from mettagrid.profiling.memory_monitor import MemoryMonitor
from mettagrid.profiling.system_monitor import SystemMonitor


class MonitoringComponent(TrainerComponent):
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

        memory_monitor, system_monitor = setup_monitoring(
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
        cleanup_monitoring(self._memory_monitor, self._system_monitor)
        self.context.memory_monitor = None
        self.context.system_monitor = None
        self._memory_monitor = None
        self._system_monitor = None
