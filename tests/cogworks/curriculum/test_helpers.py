"""Shared test utilities for curriculum tests."""

from typing import Tuple

import numpy as np
import pytest

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm


class LearningProgressTestHelper:
    """Helper utilities for learning progress algorithm tests."""

    @staticmethod
    def create_performance_sequence(base_score: float, slope: float, iterations: int) -> list[float]:
        """Create a performance sequence for testing."""
        return [base_score + i * slope for i in range(iterations)]

    @staticmethod
    def setup_fast_vs_slow_learning(
        algorithm: LearningProgressAlgorithm, task1_id: int, task2_id: int, iterations: int = 20
    ):
        """Setup fast vs slow learning comparison."""
        # Fast learning: iterations with slope 0.04
        for i in range(iterations):
            algorithm.update_task_performance(task1_id, 0.1 + i * 0.04)

        # Slow learning: iterations with slope 0.01
        for i in range(iterations):
            algorithm.update_task_performance(task2_id, 0.1 + i * 0.01)

    @staticmethod
    def setup_changing_vs_consistent_performance(
        algorithm: LearningProgressAlgorithm, task1_id: int, task2_id: int, iterations: int = 20
    ):
        """Setup changing vs consistent performance comparison."""
        # Consistent performance: exactly the same value every time
        for _ in range(iterations):
            algorithm.update_task_performance(task1_id, 0.5)

        # Changing performance: dramatic variations
        for i in range(iterations):
            if i % 3 == 0:
                algorithm.update_task_performance(task2_id, 0.9)  # High performance
            elif i % 3 == 1:
                algorithm.update_task_performance(task2_id, 0.1)  # Low performance
            else:
                algorithm.update_task_performance(task2_id, 0.5)  # Medium performance

    @staticmethod
    def setup_three_learning_patterns(
        algorithm: LearningProgressAlgorithm, task_ids: Tuple[int, int, int], iterations: int = 20
    ):
        """Setup three different learning patterns for comparison."""
        task1_id, task2_id, task3_id = task_ids

        for i in range(iterations):
            algorithm.update_task_performance(task1_id, 0.1 + i * 0.05)  # Medium improvement
            algorithm.update_task_performance(task2_id, 0.1 + i * 0.02)  # Slow improvement
            algorithm.update_task_performance(task3_id, 0.1 + i * 0.08)  # Fast improvement


class CurriculumTestHelper:
    """Helper utilities for curriculum tests."""

    @staticmethod
    def assert_step_result(result, expected):
        """Assert that step result matches expected values."""
        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=False):
            if isinstance(r, np.ndarray) and isinstance(e, np.ndarray):
                np.testing.assert_array_equal(r, e)
            else:
                assert r == e

    @staticmethod
    def create_curriculum_with_capacity(capacity: int, **kwargs):
        """Create a curriculum with specific capacity for testing."""
        from metta.cogworks.curriculum import CurriculumConfig, SingleTaskGeneratorConfig
        from metta.mettagrid.mettagrid_config import MettaGridConfig

        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=capacity, **kwargs)
        return config


class MockTaskGenerator:
    """Mock task generator for testing."""

    def get_task(self, task_id):
        return {"task_id": task_id}


@pytest.fixture
def mock_task_generator():
    """Create a mock task generator for testing."""
    return MockTaskGenerator()
