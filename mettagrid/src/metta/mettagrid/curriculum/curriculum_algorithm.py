"""
Base classes for curriculum algorithms used with Curriculum.

This module defines the abstract base classes and interfaces that all
curriculum algorithms must implement to work with the Curriculum system.
"""

import abc
from abc import ABC
from typing import List, Optional

import numpy as np

from metta.common.util.config import Config


class CurriculumAlgorithmHypers(Config, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    initial_weights: Optional[List[float]] = None

    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        """Create the curriculum algorithm with these hyperparameters.

        Args:
            num_tasks: Number of tasks the algorithm will manage

        Returns:
            Configured curriculum algorithm instance
        """
        # The default implementation is to use DiscreteRandomCurriculum
        return DiscreteRandomCurriculum(num_tasks, self)


class CurriculumAlgorithm(ABC):
    """Base class for curriculum algorithms that manage task sampling weights.

    Curriculum algorithms are responsible for:
    1. Maintaining weights for each child task
    2. Updating weights based on task completion feedback
    3. Providing normalized probabilities for sampling

    The Curriculum will use these algorithms to decide which child to sample next.
    """

    num_tasks: int
    weights: np.ndarray
    probabilities: np.ndarray
    hypers: CurriculumAlgorithmHypers

    # API that Curriculum uses

    def update(self, child_idx: int, score: float) -> None:
        """Update weights in-place based on task completion."""
        self._update_weights(child_idx, score)
        self._update_probabilities()

    def sample_idx(self) -> int:
        """Sample a child index based on current probabilities."""
        return np.random.choice(len(self.probabilities), p=self.probabilities)

    # Subclass methods to override

    def __init__(self, num_tasks: int, hypers: Optional[CurriculumAlgorithmHypers] = None):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks

        if hypers is None:
            hypers = CurriculumAlgorithmHypers()
        self.hypers = hypers

        if hypers.initial_weights is None:
            self.weights = np.ones(num_tasks, dtype=np.float32)
        else:
            self.weights = np.array(hypers.initial_weights, dtype=np.float32)
            if len(self.weights) != num_tasks:
                raise ValueError(
                    f"Initial weights must have length {num_tasks}. weights {self.weights} length: {len(self.weights)}"
                )

        self._update_probabilities()

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        return {}

    @abc.abstractmethod
    def _update_weights(self, child_idx: int, score: float) -> None:
        """Logic for updating weights in-place based on task completion goes here."""
        pass

    # Helper methods

    def _update_probabilities(self):
        """Update the probability distribution based on current weights."""
        assert len(self.weights) == self.num_tasks, (
            f"Weights must have length {self.num_tasks}. weights {self.weights} length: {len(self.weights)}"
        )
        assert self.weights.sum() > 0, f"Weights must be non-zero-sum. weights {self.weights} sum: {self.weights.sum()}"
        assert np.all(self.weights >= 0), f"Weights must be non-negative. weights {self.weights}"
        self.probabilities = self.weights / self.weights.sum()


class DiscreteRandomHypers(CurriculumAlgorithmHypers):
    """Hyperparameters for DiscreteRandomCurriculum."""

    pass


class DiscreteRandomCurriculum(CurriculumAlgorithm):
    """Curriculum algorithm that samples from a discrete distribution of weights.

    Already implemented by CurriculumAlgorithm base class - this just provides
    a named class for the simplest case where weights don't change based on
    task performance.
    """

    def _update_weights(self, child_idx: int, score: float) -> None:
        pass
