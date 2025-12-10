"""Base class for alignment metrics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class AlignmentMetric(ABC):
    """
    Base class for alignment metrics.

    All alignment metrics should:
    - Return values in [0, 1] where higher is better
    - Be computed from observable trajectories
    - Be framework-agnostic
    """

    def __init__(self, name: str):
        """
        Initialize the metric.

        Args:
            name: Human-readable name for this metric
        """
        self.name = name
        self._history: list[float] = []

    @abstractmethod
    def compute(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute the alignment metric.

        Args:
            positions: Agent positions over time, shape (T, d) where T is timesteps, d is dimension
            velocities: Agent velocities over time, shape (T, d)
            task_directions: Task direction vectors over time, shape (T, d)
            dt: Time step size
            **kwargs: Additional metric-specific parameters

        Returns:
            Metric value in [0, 1], higher is better
        """
        pass

    def update(self, value: float) -> None:
        """Update metric history."""
        self._history.append(value)

    def get_history(self) -> list[float]:
        """Get metric history."""
        return self._history

    def reset(self) -> None:
        """Reset metric history."""
        self._history = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
