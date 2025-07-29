"""System and memory monitoring utilities for training."""

import logging
from typing import Any, Optional, Tuple

from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.system_monitor import SystemMonitor
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def setup_monitoring(
    policy: Any,
    experience: Experience,
    is_master: bool,
    timer: Optional[Stopwatch] = None,
) -> Tuple[Optional[MemoryMonitor], Optional[SystemMonitor]]:
    """Set up memory and system monitoring for master rank.

    Args:
        policy: Policy model to monitor
        experience: Experience buffer to monitor
        is_master: Whether this is the master rank
        timer: Optional stopwatch timer for system monitor

    Returns:
        Tuple of (memory_monitor, system_monitor), both None if not master
    """
    if not is_master:
        return None, None

    # Create memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.add(experience, name="Experience", track_attributes=True)
    memory_monitor.add(policy, name="Policy", track_attributes=False)

    # Create system monitor
    system_monitor = SystemMonitor(
        sampling_interval_sec=1.0,
        history_size=100,
        logger=logger,
        auto_start=True,
        external_timer=timer,
    )

    logger.info("Initialized memory and system monitoring")
    return memory_monitor, system_monitor


def cleanup_monitoring(
    memory_monitor: Optional[MemoryMonitor],
    system_monitor: Optional[SystemMonitor],
) -> None:
    """Clean up monitoring resources.

    Args:
        memory_monitor: Memory monitor to clean up
        system_monitor: System monitor to stop
    """
    if memory_monitor:
        memory_monitor.clear()
        logger.debug("Cleared memory monitor")

    if system_monitor:
        system_monitor.stop()
        logger.debug("Stopped system monitor")


def get_memory_stats(memory_monitor: Optional[MemoryMonitor]) -> dict:
    """Get current memory statistics.

    Args:
        memory_monitor: Memory monitor instance

    Returns:
        Dictionary of memory stats, empty if monitor is None
    """
    if not memory_monitor:
        return {}

    stats = {}
    for name, info in memory_monitor.get_all_stats().items():
        stats[f"memory/{name}_mb"] = info["size_mb"]
        if "attributes" in info:
            for attr_name, attr_size in info["attributes"].items():
                stats[f"memory/{name}_{attr_name}_mb"] = attr_size

    return stats


def get_system_stats(system_monitor: Optional[SystemMonitor]) -> dict:
    """Get current system statistics.

    Args:
        system_monitor: System monitor instance

    Returns:
        Dictionary of system stats, empty if monitor is None
    """
    if not system_monitor:
        return {}

    current_stats = system_monitor.get_current_stats()
    stats = {}

    if current_stats:
        stats.update(
            {
                "system/cpu_percent": current_stats.get("cpu_percent", 0),
                "system/memory_percent": current_stats.get("memory_percent", 0),
                "system/memory_used_gb": current_stats.get("memory_used_gb", 0),
            }
        )

        # Add GPU stats if available
        gpu_stats = current_stats.get("gpu", {})
        for gpu_id, gpu_data in gpu_stats.items():
            stats.update(
                {
                    f"system/gpu_{gpu_id}_utilization": gpu_data.get("utilization", 0),
                    f"system/gpu_{gpu_id}_memory_percent": gpu_data.get("memory_percent", 0),
                    f"system/gpu_{gpu_id}_temperature": gpu_data.get("temperature", 0),
                }
            )

    return stats
