"""Base task interface for alignment metrics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class TaskInterface(ABC):
    """
    Base interface for task specifications.

    Tasks define what agents should do and provide the task direction field g(x,t)
    that indicates "what counts as progress" at each point in space and time.
    """

    @abstractmethod
    def get_task_direction(self, position: npt.NDArray[np.floating[Any]], time: float) -> npt.NDArray[np.floating[Any]]:
        """
        Get the task direction at a given position and time.

        Args:
            position: Current position, shape (d,)
            time: Current time

        Returns:
            Unit task direction vector, shape (d,)
        """
        pass

    @abstractmethod
    def is_complete(self, position: npt.NDArray[np.floating[Any]], time: float) -> bool:
        """
        Check if the task is complete at the given position and time.

        Args:
            position: Current position, shape (d,)
            time: Current time

        Returns:
            True if task is complete
        """
        pass
