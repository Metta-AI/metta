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
        """Test that tasks are only evicted after minimum presentations."""
        arena = make_arena(num_agents=4)
        bucketed_config = cc.bucketed(arena)
        bucketed_config.add_bucket("game.agent.rewards.inventory.ore_red", [0.0, 1.0])

        config = bucketed_config.to_curriculum()
        config.num_active_tasks = 3  # Very small for testing
        config.min_presentations_for_eviction = 5  # Require 5 presentations
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1, exploration_bonus=0.1, max_memory_tasks=100
        )

        curriculum = config.make()

        # Track presentations to verify minimum requirement logic
        task_presentations = {}

        # Give presentations ensuring we track the requirements correctly
        for _ in range(12):  # Spread presentations across tasks
            task = curriculum.get_task()
            task_id = task._task_id
            task_presentations[task_id] = task_presentations.get(task_id, 0) + 1

            task.complete(0.1)
            curriculum.update_task_performance(task_id, 0.1)

            # Check if any task has reached the minimum presentations
            max_presentations = max(task_presentations.values())
            if max_presentations >= config.min_presentations_for_eviction:
                break

        # Verify that a task with sufficient presentations can be evicted
        algorithm = curriculum._algorithm
        evictable_tasks = [
            tid
            for tid in curriculum._tasks.keys()
            if algorithm.should_evict_task(tid, config.min_presentations_for_eviction)
        ]

        # Check if any task has enough presentations to be considered for eviction
        max_presentations = max(task_presentations.values()) if task_presentations else 0

        if max_presentations >= config.min_presentations_for_eviction:
            # Some task should be evictable
            assert len(evictable_tasks) > 0
        else:
            # No task should be evictable yet
            assert len(evictable_tasks) == 0

        # Continue to ensure eviction happens when requirements are met
        for _ in range(10):  # More episodes to ensure some tasks get enough presentations
            task = curriculum.get_task()
            task.complete(0.1)
            curriculum.update_task_performance(task._task_id, 0.1)

        final_stats = curriculum.stats()

        # Should have evictions once tasks meet the presentation requirement
        assert final_stats["num_evicted"] > 0

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
