"""
Path Efficiency Metric (E_i).

Measures the straightness of the realized path.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.metrics.base import AlignmentMetric


class PathEfficiencyMetric(AlignmentMetric):
    """
    Path Efficiency Metric (E_i).

    Measures straightness of the realized path:
        E_i = displacement / path_length

    where:
        - displacement = ||x_i(T) - x_i(0)||
        - path_length = ∫ ||v_i(t)|| dt

    E_i = 1 for geodesic-like motion, falls with detours or loops.
    Invariant to timing but sensitive to obstacles.
    """

    def __init__(self):
        """Initialize the Path Efficiency metric."""
        super().__init__(name="Path Efficiency")

    def compute(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute path efficiency for an agent.

        Args:
            positions: Agent positions over time, shape (T, d)
            velocities: Agent velocities over time, shape (T, d)
            task_directions: Task direction vectors (not used for this metric)
            dt: Time step size

        Returns:
            E_i ∈ [0, 1], higher means straighter path
        """
        if len(positions) < 2:
            return 1.0  # No path taken

        # Compute displacement
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))

        # Compute path length
        speeds = np.linalg.norm(velocities, axis=1)
        path_length = float(np.sum(speeds) * dt)

        # Handle edge cases
        if path_length < 1e-8:
            # No movement
            if displacement < 1e-8:
                return 1.0  # Stayed at goal
            else:
                return 0.0  # Teleported?

        # Compute efficiency
        E_i = displacement / path_length

        # Clip to [0, 1] to handle numerical errors
        E_i = min(1.0, max(0.0, E_i))

        return E_i

    def compute_loopiness(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        dt: float,
    ) -> float:
        """
        Compute loopiness metric (misalignment detector).

        Λ = (path_length - displacement) / path_length

        Higher values indicate more detours and loops.

        Args:
            positions: Shape (T, d)
            velocities: Shape (T, d)
            dt: Time step size

        Returns:
            Loopiness ∈ [0, 1], higher means more loops
        """
        if len(positions) < 2:
            return 0.0

        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        speeds = np.linalg.norm(velocities, axis=1)
        path_length = float(np.sum(speeds) * dt)

        eps = 1e-8
        loopiness = (path_length - displacement) / (path_length + eps)

        return float(np.clip(loopiness, 0.0, 1.0))
