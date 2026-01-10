"""Setpoint task: reach a goal position."""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.task_interfaces.base import TaskInterface


class SetpointTask(TaskInterface):
    """
    Setpoint task: reach a goal position or set.

    The task direction g(x,t) points toward the closest goal.

    Args:
        goal: Goal position(s), shape (d,) or (N, d) for multiple goals
        tolerance: Distance threshold for task completion
    """

    def __init__(
        self,
        goal: npt.NDArray[np.floating[Any]],
        tolerance: float = 0.1,
    ):
        """Initialize setpoint task."""
        self.goal = np.atleast_2d(goal)  # Ensure 2D for multiple goals
        self.tolerance = tolerance

    def get_task_direction(self, position: npt.NDArray[np.floating[Any]], time: float) -> npt.NDArray[np.floating[Any]]:
        """
        Get direction to closest goal.

        Args:
            position: Current position, shape (d,)
            time: Current time (not used for static goals)

        Returns:
            Unit direction vector toward closest goal
        """
        # Find closest goal
        if len(self.goal) == 1:
            closest_goal = self.goal[0]
        else:
            distances = np.linalg.norm(self.goal - position, axis=1)
            closest_idx = np.argmin(distances)
            closest_goal = self.goal[closest_idx]

        # Compute direction
        direction = closest_goal - position
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            # Already at goal
            return np.zeros_like(position)

        # Return unit vector
        return direction / distance

    def is_complete(self, position: npt.NDArray[np.floating[Any]], time: float) -> bool:
        """
        Check if agent is within tolerance of any goal.

        Args:
            position: Current position
            time: Current time

        Returns:
            True if within tolerance of a goal
        """
        distances = np.linalg.norm(self.goal - position, axis=1)
        return bool(np.any(distances < self.tolerance))

    def get_closest_goal(self, position: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Get the closest goal to the given position."""
        if len(self.goal) == 1:
            return self.goal[0]

        distances = np.linalg.norm(self.goal - position, axis=1)
        closest_idx = np.argmin(distances)
        return self.goal[closest_idx]
