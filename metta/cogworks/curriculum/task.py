"""Task class for cogworks curriculum system."""

from __future__ import annotations

from metta.mettagrid.mettagrid_config import MettaGridConfig


class Task:
    """Simple task class for backward compatibility with existing curriculum code."""

    def __init__(self, task_id: str, env_cfg: MettaGridConfig):
        self._task_id = task_id
        self._env_cfg = env_cfg

    @property
    def env_cfg(self) -> MettaGridConfig:
        """Get the environment configuration."""
        return self._env_cfg

    def get_mg_config(self) -> MettaGridConfig:
        """Get the environment configuration (alternative method name)."""
        return self._env_cfg

    def get_id(self) -> str:
        """Get the task ID."""
        return self._task_id

    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task(id={self._task_id})"
