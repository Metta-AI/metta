"""
Directional Intent Metric (D_i).

Measures how persistently an agent's velocity points along the task direction.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from metta.alignment.metrics.base import AlignmentMetric


class DirectionalIntentMetric(AlignmentMetric):
    """
    Directional Intent Metric (D_i).

    Measures how persistently velocity points along the task direction:
        D_i = (1/T) ∫ max{0, ρ_i(t) - τ} dt

    where ρ_i(t) = <v_i(t), g_i(t)> / (||v_i(t)|| ||g_i(t)||) is the progress cosine.

    High values indicate sustained, purposeful motion toward the task.
    Low values indicate dithering, orbiting, or adversarial motion.

    Args:
        tolerance: Tolerance τ ∈ [0, 0.1] to suppress penalization from small heading noise
    """

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize the Directional Intent metric.

        Args:
            tolerance: Tolerance for small heading deviations (default 0.05)
        """
        super().__init__(name="Directional Intent")
        if not 0 <= tolerance <= 0.1:
            raise ValueError(f"Tolerance must be in [0, 0.1], got {tolerance}")
        self.tolerance = tolerance

    def compute(
        self,
        positions: npt.NDArray[np.floating[Any]],
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute directional intent for an agent.

        Args:
            positions: Agent positions over time, shape (T, d)
            velocities: Agent velocities over time, shape (T, d)
            task_directions: Task direction vectors over time, shape (T, d)
            dt: Time step size

        Returns:
            D_i ∈ [0, 1], higher means better alignment with task direction
        """
        T = len(velocities)
        if T == 0:
            return 0.0

        # Compute progress cosine ρ_i(t) at each timestep
        progress_cosines = self._compute_progress_cosines(velocities, task_directions)

        # Apply tolerance and clip to [0, 1]
        aligned_progress = np.maximum(0, progress_cosines - self.tolerance)

        # Time-average
        D_i = float(np.mean(aligned_progress))

        return D_i

    def _compute_progress_cosines(
        self,
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Compute progress cosine ρ_i(t) = <v, g> / (||v|| ||g||).

        Args:
            velocities: Shape (T, d)
            task_directions: Shape (T, d)

        Returns:
            Progress cosines, shape (T,)
        """
        # Compute norms
        v_norms = np.linalg.norm(velocities, axis=1)
        g_norms = np.linalg.norm(task_directions, axis=1)

        # Compute dot products
        dot_products = np.sum(velocities * task_directions, axis=1)

        # Avoid division by zero
        eps = 1e-8
        denominators = v_norms * g_norms + eps

        # Compute cosines
        progress_cosines = dot_products / denominators

        # Clip to [-1, 1] to handle numerical errors
        progress_cosines = np.clip(progress_cosines, -1.0, 1.0)

        return progress_cosines

    def compute_anti_progress_mass(
        self,
        velocities: npt.NDArray[np.floating[Any]],
        task_directions: npt.NDArray[np.floating[Any]],
        dt: float,
    ) -> float:
        """
        Compute anti-progress mass (misalignment detector).

        A_i = ∫ v_i(t) · max{0, τ - ρ_i(t)} dt

        This accumulates how much and how long the agent moves against the task.

        Args:
            velocities: Shape (T, d)
            task_directions: Shape (T, d)
            dt: Time step size

        Returns:
            Anti-progress mass (higher means more misalignment)
        """
        progress_cosines = self._compute_progress_cosines(velocities, task_directions)
        speeds = np.linalg.norm(velocities, axis=1)

        # Accumulate anti-progress
        anti_progress = speeds * np.maximum(0, self.tolerance - progress_cosines)

        # Integrate over time
        A_i = float(np.sum(anti_progress) * dt)

        return A_i
