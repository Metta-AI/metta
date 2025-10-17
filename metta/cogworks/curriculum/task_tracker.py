"""Backward compatibility shim for task_tracker.

DEPRECATED: Use 'from agora.tracking import TaskTracker' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.task_tracker is deprecated. Use 'from agora.tracking import TaskTracker' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agora.tracking import (  # noqa: E402
    CentralizedTaskTracker,
    LocalMemoryBackend,
    LocalTaskTracker,
    SharedMemoryBackend,
    TaskMemoryBackend,
    TaskTracker,
)

__all__ = [
    "TaskTracker",
    "TaskMemoryBackend",
    "LocalMemoryBackend",
    "SharedMemoryBackend",
    "LocalTaskTracker",
    "CentralizedTaskTracker",
]
