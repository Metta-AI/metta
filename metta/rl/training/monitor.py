"""Trainer component responsible for setting up system and memory monitoring."""

import logging
import typing

import metta.rl.training
import mettagrid.profiling.memory_monitor
import mettagrid.profiling.system_monitor

logger = logging.getLogger(__name__)


class Monitor(metta.rl.training.TrainerComponent):
    """Manage memory and system monitors independently of stats reporting."""

    _master_only = True

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__()
        self._enabled = enabled
        self._memory_monitor: typing.Optional[mettagrid.profiling.memory_monitor.MemoryMonitor] = None
        self._system_monitor: typing.Optional[mettagrid.profiling.system_monitor.SystemMonitor] = None

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._enabled:
            return

        memory_monitor, system_monitor = self._setup(
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
        self._cleanup(self._memory_monitor, self._system_monitor)
        self.context.memory_monitor = None
        self.context.system_monitor = None
        self._memory_monitor = None
        self._system_monitor = None

    @staticmethod
    def _setup(
        *, policy, experience, timer
    ) -> tuple[mettagrid.profiling.memory_monitor.MemoryMonitor, mettagrid.profiling.system_monitor.SystemMonitor]:
        memory_monitor = mettagrid.profiling.memory_monitor.MemoryMonitor()
        memory_monitor.add(experience, name="Experience", track_attributes=True)
        memory_monitor.add(policy, name="Policy", track_attributes=False)

        system_monitor = mettagrid.profiling.system_monitor.SystemMonitor(
            sampling_interval_sec=1.0,
            history_size=100,
            log_level=logger.getEffectiveLevel(),
            auto_start=True,
            external_timer=timer,
        )

        logger.info("Initialized memory and system monitoring")
        return memory_monitor, system_monitor

    @staticmethod
    def _cleanup(
        memory_monitor: mettagrid.profiling.memory_monitor.MemoryMonitor | None,
        system_monitor: mettagrid.profiling.system_monitor.SystemMonitor | None,
    ) -> None:
        if memory_monitor:
            memory_monitor.clear()
            logger.debug("Cleared memory monitor")

        if system_monitor:
            system_monitor.stop()
            logger.debug("Stopped system monitor")
