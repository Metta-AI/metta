"""Test that num_active_tasks correctly reports count from shared memory.

This test verifies the fix for the critical bug where num_active_tasks was reporting
the count from the local self._tasks dict instead of the shared memory TaskTracker,
causing all downstream Gini calculations to be incorrect.
"""

import uuid

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig, CurriculumTask
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from mettagrid.config.mettagrid_config import MettaGridConfig


class TestTaskCountReporting:
    """Test that task counts are correctly reported from shared memory."""

    def test_num_active_tasks_from_shared_memory(self):
        """Test that num_active_tasks reports the correct count from TaskTracker when using shared memory."""
        # Create curriculum with shared memory enabled
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        # Use short session_id to avoid POSIX shared memory name length limits (~31 chars)
        session_id = uuid.uuid4().hex[:8]
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            use_shared_memory=True,
            session_id=session_id,
        )
        config = CurriculumConfig(
            task_generator=task_gen,
            algorithm_config=algorithm_config,
            num_active_tasks=10,
            seed=42,
            defer_init=True,
        )
        curriculum = Curriculum(config)

        # Manually add tasks to TaskTracker (simulating multiple workers)
        for i in range(10):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            curriculum._algorithm.on_task_created(task)

        # Simulate that only 2 tasks are in the local curriculum._tasks dict
        # (as would happen with multiple workers where each worker only sees a subset)
        curriculum._tasks = {0: task, 1: task}  # Only 2 tasks locally

        # Get stats - should report 10 tasks from TaskTracker, not 2 from local dict
        stats = curriculum.get_base_stats()

        assert stats["num_active_tasks"] == 10.0, (
            f"Expected 10 tasks from TaskTracker, got {stats['num_active_tasks']} "
            f"(local dict has {len(curriculum._tasks)} tasks)"
        )

    def test_num_active_tasks_without_shared_memory(self):
        """Test that num_active_tasks falls back to local dict count when not using shared memory."""
        # Create curriculum without shared memory
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen,
            algorithm_config=algorithm_config,
            num_active_tasks=10,
            seed=42,
            defer_init=True,
        )
        curriculum = Curriculum(config)

        # Add 5 tasks to local dict
        for i in range(5):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            curriculum._tasks[i] = task
            curriculum._algorithm.on_task_created(task)

        # Get stats - should report 5 tasks from local dict
        stats = curriculum.get_base_stats()

        assert stats["num_active_tasks"] == 5.0, f"Expected 5 tasks from local dict, got {stats['num_active_tasks']}"

    def test_num_active_tasks_without_algorithm(self):
        """Test that num_active_tasks uses local dict when no algorithm is present."""
        # Create curriculum without algorithm
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(
            task_generator=task_gen,
            algorithm_config=None,  # No algorithm
            num_active_tasks=10,
            seed=42,
            defer_init=True,
        )
        curriculum = Curriculum(config)

        # Add 3 tasks to local dict
        for i in range(3):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            curriculum._tasks[i] = task

        # Get stats - should report 3 tasks from local dict
        stats = curriculum.get_base_stats()

        assert stats["num_active_tasks"] == 3.0, f"Expected 3 tasks from local dict, got {stats['num_active_tasks']}"
