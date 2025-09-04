import logging
from typing import Any, Tuple

from metta.mettagrid.profiling.memory_monitor import MemoryMonitor
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.mettagrid.profiling.system_monitor import SystemMonitor
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def setup_monitoring(
    policy: Any,
    experience: Experience,
    timer: Stopwatch | None = None,
) -> Tuple[MemoryMonitor, SystemMonitor]:
    """Set up memory and system monitoring (should only be called on master rank)."""

    # Create memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.add(experience, name="Experience", track_attributes=True)
    memory_monitor.add(policy, name="Policy", track_attributes=False)

    # Create system monitor
    system_monitor = SystemMonitor(
        sampling_interval_sec=1.0,
        history_size=100,
        log_level=logger.getEffectiveLevel(),
        auto_start=True,
        external_timer=timer,
    )

    logger.info("Initialized memory and system monitoring")
    return memory_monitor, system_monitor


def cleanup_monitoring(
    memory_monitor: MemoryMonitor | None,
    system_monitor: SystemMonitor | None,
) -> None:
    """Clean up memory and system monitoring resources."""
    if memory_monitor:
        memory_monitor.clear()
        logger.debug("Cleared memory monitor")

    if system_monitor:
        system_monitor.stop()
        logger.debug("Stopped system monitor")
