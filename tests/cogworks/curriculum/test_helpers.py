"""Shared test utilities for curriculum tests."""

from typing import Tuple, Union

import numpy as np
import pytest

import softmax.cogworks.curriculum as cc
from softmax.cogworks.curriculum.curriculum import CurriculumConfig
from softmax.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig


class CurriculumTestHelper:
    """Unified helper utilities for all curriculum tests."""

    @staticmethod
    def create_performance_sequence(base_score: float, slope: float, iterations: int) -> list[float]:
        """Create a performance sequence for testing."""
        return [base_score + i * slope for i in range(iterations)]

    @staticmethod
    def setup_learning_comparison(
        algorithm: LearningProgressAlgorithm,
        task_ids: Union[int, Tuple[int, ...]],
        pattern: str = "fast_vs_slow",
        iterations: int = 20,
    ) -> None:
        """Setup learning comparison with different patterns.

        Args:
            algorithm: Learning progress algorithm instance
            task_ids: Single task ID or tuple of task IDs
            pattern: Comparison pattern ('fast_vs_slow', 'changing_vs_consistent', 'three_patterns')
            iterations: Number of iterations to run
        """
        if isinstance(task_ids, int):
            task_ids = (task_ids,)

        if pattern == "fast_vs_slow" and len(task_ids) >= 2:
            # Fast learning: iterations with slope 0.04
            for i in range(iterations):
                algorithm.update_task_performance(task_ids[0], 0.1 + i * 0.04)
            # Slow learning: iterations with slope 0.01
            for i in range(iterations):
                algorithm.update_task_performance(task_ids[1], 0.1 + i * 0.01)

        elif pattern == "changing_vs_consistent" and len(task_ids) >= 2:
            # Consistent performance: exactly the same value every time
            for _ in range(iterations):
                algorithm.update_task_performance(task_ids[0], 0.5)
            # Changing performance: dramatic variations
            for i in range(iterations):
                if i % 3 == 0:
                    algorithm.update_task_performance(task_ids[1], 0.9)  # High performance
                elif i % 3 == 1:
                    algorithm.update_task_performance(task_ids[1], 0.1)  # Low performance
                else:
                    algorithm.update_task_performance(task_ids[1], 0.5)  # Medium performance

        elif pattern == "three_patterns" and len(task_ids) >= 3:
            for i in range(iterations):
                algorithm.update_task_performance(task_ids[0], 0.1 + i * 0.05)  # Medium improvement
                algorithm.update_task_performance(task_ids[1], 0.1 + i * 0.02)  # Slow improvement
                algorithm.update_task_performance(task_ids[2], 0.1 + i * 0.08)  # Fast improvement

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
        from softmax.cogworks.curriculum import SingleTaskGenerator
        from mettagrid.config.mettagrid_config import MettaGridConfig

        task_gen_config = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=capacity, **kwargs)
        return config

    @staticmethod
    def create_test_curriculum(curriculum_type: str = "basic", **kwargs):
        """Create a test curriculum of specified type.

        Args:
            curriculum_type: Type of curriculum ('basic', 'with_algorithm', 'production')
            **kwargs: Additional configuration parameters
        """
        from softmax.cogworks.curriculum import SingleTaskGenerator
        from mettagrid.config.mettagrid_config import MettaGridConfig

        base_config = SingleTaskGenerator.Config(env=MettaGridConfig())

        if curriculum_type == "basic":
            return CurriculumConfig(task_generator=base_config, **kwargs)
        elif curriculum_type == "with_algorithm":
            algorithm = LearningProgressConfig(**kwargs.get("algorithm_params", {}))
            return CurriculumConfig(task_generator=base_config, algorithm_config=algorithm, **kwargs)
        elif curriculum_type == "production":
            # Create production-like curriculum with buckets
            tasks = cc.bucketed(MettaGridConfig())
            tasks.add_bucket("test.param", [1, 2, 3])
            return tasks.to_curriculum(**kwargs)
        else:
            raise ValueError(f"Unknown curriculum type: {curriculum_type}")


class MockTaskGenerator:
    """Mock task generator for testing."""

    def get_task(self, task_id):
        return {"task_id": task_id}


# Backward compatibility aliases
LearningProgressTestHelper = CurriculumTestHelper


@pytest.fixture
def mock_task_generator():
    """Create a mock task generator for testing."""
    return MockTaskGenerator()


@pytest.fixture
def curriculum_helper():
    """Create a curriculum test helper instance."""
    return CurriculumTestHelper()
