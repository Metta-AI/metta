"""
Goal Attainment Metric (A_i).

Measures how close the agent finishes to the task target.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.metrics.base import AlignmentMetric


class GoalAttainmentMetric(AlignmentMetric):
    """
    Goal Attainment Metric (A_i).

    Measures how close the agent finishes to the task target:
        A_i = exp(-d(x_i(T), G(T)) / ℓ)

    where:
        - d(·,·) is distance to goal set G
        - ℓ is a characteristic scale (tolerance)

    High values mean the agent ended up where it should.
    Low values mean residual task error.

    Args:
        scale: Characteristic scale ℓ for acceptable tolerance
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize the Goal Attainment metric.

        Args:
            scale: Characteristic scale for distance normalization (default 1.0)
        """
        super().__init__(name="Goal Attainment")
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        self.scale = scale

    def compute(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        goal: npt.NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute goal attainment for an agent.

        Args:
            positions: Agent positions over time, shape (T, d)
            velocities: Agent velocities (not used for this metric)
            task_directions: Task directions (not used for this metric)
            dt: Time step size (not used for this metric)
            goal: Goal position, shape (d,). If None, uses last task direction

        Returns:
            A_i ∈ [0, 1], higher means closer to goal
        """
        if len(positions) == 0:
            return 0.0

        final_position = positions[-1]

        # Determine goal
        if goal is None:
            if len(task_directions) > 0:
                # Use task direction as proxy for goal direction
                # Assume goal is in the direction of the task
                goal = final_position + task_directions[-1]
            else:
                # No goal information, assume perfect attainment
                return 1.0

        # Compute distance to goal
        distance = float(np.linalg.norm(final_position - goal))

        # Exponential decay with scale
        A_i = np.exp(-distance / self.scale)

        return float(A_i)

    def compute_with_goal_set(
        self,
        final_position: npt.NDArray[np.floating[Any]],
        goal_set: npt.NDArray[np.floating[Any]],
    ) -> float:
        """
        Compute attainment for a goal set (multiple possible goals).

        Args:
            final_position: Final agent position, shape (d,)
            goal_set: Set of goal positions, shape (N, d)

        Returns:
            A_i based on distance to closest goal
        """
        if len(goal_set) == 0:
            return 0.0

        # Find closest goal
        distances = np.linalg.norm(goal_set - final_position, axis=1)
        min_distance = float(np.min(distances))

        # Exponential decay
        A_i = np.exp(-min_distance / self.scale)

        return float(A_i)
