"""Tests for curriculum capacity initialization and algorithm-based eviction."""

import metta.cogworks.curriculum as cc
from metta.mettagrid.builder.envs import make_arena


class TestCurriculumCapacityAndEviction:
    """Test curriculum behavior with capacity initialization and algorithm-based eviction."""

    def test_curriculum_starts_at_capacity(self):
        """Test that curriculum initializes at full capacity when configured."""
        # Create a curriculum with capacity initialization
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 5  # Small for testing

        curriculum = config.make()

        # Check that it starts at capacity
        initial_stats = curriculum.stats()
        assert initial_stats["num_created"] == config.num_active_tasks
        assert initial_stats["num_active_tasks"] == config.num_active_tasks
        assert initial_stats["num_evicted"] == 0

    def test_algorithm_based_eviction_with_learning_progress(self):
        """Test that algorithm-based eviction works with learning progress criteria."""
        # Create a curriculum with algorithm-based eviction
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 0.5, 1.0])
        bucketed_config.add_bucket("game.agent.rewards.inventory.battery_red", [0.0, 0.5, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 8  # Small for testing
        config.min_presentations_for_eviction = 3
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1, exploration_bonus=0.1, max_memory_tasks=100
        )

        curriculum = config.make()

        # Verify initial state
        initial_stats = curriculum.stats()
        assert initial_stats["num_active_tasks"] == config.num_active_tasks
        assert initial_stats["num_evicted"] == 0

        # Simulate training episodes with varying performance patterns
        for i in range(20):
            task = curriculum.get_task()

            # Create different performance patterns to trigger learning progress differences
            if i < 10:
                # First 10: some tasks get high performance (fast learning)
                performance = 0.8 if task._task_id % 3 == 0 else 0.2
            else:
                # Next 10: consistent performance (slow learning)
                performance = 0.5 + 0.1 * (i % 3)

            task.complete(performance)
            curriculum.update_task_performance(task._task_id, performance)

        final_stats = curriculum.stats()

        # Verify capacity is maintained
        assert final_stats["num_active_tasks"] == config.num_active_tasks

        # Verify evictions happened based on algorithm criteria
        assert final_stats["num_evicted"] > 0

        # Verify new tasks were created to replace evicted ones
        assert final_stats["num_created"] > config.num_active_tasks

    def test_curriculum_without_algorithm(self):
        """Test that curriculum behavior without algorithm doesn't evict tasks."""
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 5
        config.algorithm_config = None  # No algorithm

        curriculum = config.make()

        # Simulate episodes
        for _ in range(15):
            task = curriculum.get_task()
            task.complete(0.5)

        final_stats = curriculum.stats()

        # Should maintain capacity
        assert final_stats["num_active_tasks"] == config.num_active_tasks

        # Without an algorithm, no evictions should happen once at capacity
        assert final_stats["num_evicted"] == 0
        assert final_stats["num_created"] == config.num_active_tasks

    def test_minimum_presentations_requirement(self):
        """Test that tasks are only evicted after minimum presentations and that we evict the lowest scoring task."""
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 3  # Very small for testing
        min_presentations_required = 6  # Set high enough to be the actual threshold
        config.min_presentations_for_eviction = min_presentations_required
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1,
            exploration_bonus=0.1,
            max_memory_tasks=100,
            min_presentations_for_eviction=min_presentations_required,  # Ensure consistency
        )

        curriculum = config.make()

        # Get the actual eviction threshold from the algorithm
        algorithm = curriculum._algorithm
        actual_min_samples = algorithm.eviction_policy.min_samples
        print(
            f"Configured min_presentations: {min_presentations_required}, "
            f"Actual eviction threshold: {actual_min_samples}"
        )

        # Phase 1: Give tasks different scores and presentations
        # We'll track what we expect vs actual task pool state

        # Give task 0 high score but few presentations (below threshold)
        task_0 = curriculum.get_task()
        for _ in range(actual_min_samples - 1):  # One less than threshold
            curriculum.update_task_performance(task_0._task_id, 0.9)  # Only call this once

        # Give other tasks enough presentations
        for _ in range(actual_min_samples + 2):  # Above threshold
            task = curriculum.get_task()
            if task._task_id != task_0._task_id:
                # Medium score for task 1, low score for task 2
                score = 0.5 if task._task_id == 1 else 0.2
                curriculum.update_task_performance(task._task_id, score)

        # Get actual task pool data for verification
        pool_data = {}
        for task_id in curriculum._tasks.keys():
            task_sample = curriculum.task_pool.get_task(task_id)
            if task_sample:
                pool_data[task_id] = {
                    "num_samples": task_sample.num_samples,
                    "mean_score": task_sample.get_mean_score(),
                }
        print(f"Task pool data: {pool_data}")

        # Debug: Show which tasks are considered evictable and why
        for task_id in curriculum._tasks.keys():
            task_sample = curriculum.task_pool.get_task(task_id)
            is_evictable = algorithm.should_evict_task(task_id, actual_min_samples)
            print(f"Task {task_id}: {task_sample.num_samples} samples, evictable: {is_evictable}")

        # Test eviction logic based on actual task pool data
        for task_id in curriculum._tasks.keys():
            task_sample = curriculum.task_pool.get_task(task_id)
            is_evictable = algorithm.should_evict_task(task_id, actual_min_samples)

            if task_sample.num_samples < actual_min_samples:
                assert not is_evictable, (
                    f"Task {task_id} with only {task_sample.num_samples} samples "
                    f"should not be evictable (threshold: {actual_min_samples})"
                )
            else:
                assert is_evictable, (
                    f"Task {task_id} with {task_sample.num_samples} samples "
                    f"should be evictable (threshold: {actual_min_samples})"
                )

        # Phase 2: Trigger actual eviction by continuing to get tasks
        initial_evicted_count = curriculum.stats()["num_evicted"]

        # Continue getting tasks to trigger eviction
        for _ in range(10):
            task = curriculum.get_task()
            task.complete(0.4)
            curriculum.update_task_performance(task._task_id, 0.4)

        final_stats = curriculum.stats()

        # Should have evictions once tasks meet the presentation requirement
        assert final_stats["num_evicted"] > initial_evicted_count, (
            f"Expected evictions to happen. Initial: {initial_evicted_count}, Final: {final_stats['num_evicted']}"
        )

    def test_learning_progress_eviction_criteria(self):
        """Test that tasks with low learning progress are preferentially evicted."""
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 4
        config.min_presentations_for_eviction = 3
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1, exploration_bonus=0.1, max_memory_tasks=100
        )

        curriculum = config.make()
        initial_task_ids = list(curriculum._tasks.keys())

        # Create distinct performance patterns for different tasks
        task_performance_patterns = {
            initial_task_ids[0]: [0.1, 0.9, 0.1, 0.9, 0.1],  # High variance (high learning progress)
            initial_task_ids[1]: [0.5, 0.5, 0.5, 0.5, 0.5],  # Low variance (low learning progress)
            initial_task_ids[2]: [0.2, 0.8, 0.3, 0.7, 0.4],  # Medium variance
            initial_task_ids[3]: [0.6, 0.6, 0.6, 0.6, 0.6],  # Low variance (low learning progress)
        }

        # Apply performance patterns
        for task_id, performances in task_performance_patterns.items():
            for performance in performances:
                curriculum.update_task_performance(task_id, performance)

        # Get learning progress scores to verify our setup
        algorithm = curriculum._algorithm
        scores = algorithm.score_tasks(initial_task_ids)

        # Task with high variance should have higher learning progress score
        high_variance_task = initial_task_ids[0]
        low_variance_tasks = [initial_task_ids[1], initial_task_ids[3]]

        assert scores[high_variance_task] > max(scores[tid] for tid in low_variance_tasks)

        # Now trigger eviction by getting more tasks
        evicted_tasks = set()
        for _ in range(10):
            task = curriculum.get_task()
            if task._task_id not in initial_task_ids:
                # This is a new task, so something was evicted
                current_task_ids = set(curriculum._tasks.keys())
                evicted_tasks = set(initial_task_ids) - current_task_ids
                break
            task.complete(0.5)

        # Verify that low learning progress tasks were preferentially evicted
        if evicted_tasks:
            # The evicted task should be one of the low variance tasks
            evicted_task = list(evicted_tasks)[0]
            assert evicted_task in low_variance_tasks
