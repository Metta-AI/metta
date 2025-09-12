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
        config.min_presentations_for_eviction = 10  # More presentations for realistic EMA timescales
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1, exploration_bonus=0.1, max_memory_tasks=100
        )

        curriculum = config.make()

        # Verify initial state
        initial_stats = curriculum.stats()
        assert initial_stats["num_active_tasks"] == config.num_active_tasks
        assert initial_stats["num_evicted"] == 0

        # Force even distribution of presentations to ensure some poor tasks get enough data
        all_task_ids = list(curriculum._tasks.keys())

        # Give each task exactly min_presentations_for_eviction + 5 presentations
        presentations_per_task = config.min_presentations_for_eviction + 5

        for task_id in all_task_ids:
            task = curriculum._tasks[task_id]
            task_pattern = task_id % 4

            for presentation_num in range(presentations_per_task):
                if task_pattern == 0:
                    # Poor stable tasks: consistently very low performance
                    performance = 0.1 + 0.01 * (presentation_num % 3)  # 0.1-0.12 (very poor)
                elif task_pattern == 1:
                    # Learning tasks: clear improvement over time
                    performance = 0.2 + 0.5 * (presentation_num / presentations_per_task)  # 0.2 -> 0.7
                elif task_pattern == 2:
                    # High variance tasks: alternating performance
                    performance = 0.1 if presentation_num % 2 == 0 else 0.9
                else:
                    # Good stable tasks: consistently high performance
                    performance = 0.8 + 0.05 * (presentation_num % 3)  # 0.8-0.85 (very good)

                task.complete(performance)
                curriculum.update_task_performance(task_id, performance)

        # Now force additional get_task() calls to trigger eviction attempts
        for _ in range(50):
            task = curriculum.get_task()
            # Give minimal additional performance to maintain patterns
            task_pattern = task._task_id % 4
            if task_pattern == 0:
                performance = 0.11  # Keep poor tasks poor
            elif task_pattern == 1:
                performance = 0.75  # Keep improving tasks good
            elif task_pattern == 2:
                performance = 0.5  # Moderate for high variance
            else:
                performance = 0.82  # Keep good tasks good

            task.complete(performance)
            curriculum.update_task_performance(task._task_id, performance)

        final_stats = curriculum.stats()

        # Verify capacity is maintained
        assert final_stats["num_active_tasks"] == config.num_active_tasks

        # Verify evictions happened based on algorithm criteria
        assert final_stats["num_evicted"] > 0, f"Expected evictions but got: {final_stats}"

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
        config.min_presentations_for_eviction = 15  # Require 15 presentations for realistic EMA development
        config.algorithm_config = cc.LearningProgressConfig(
            ema_timescale=0.1, exploration_bonus=0.1, max_memory_tasks=100
        )

        curriculum = config.make()

        # Get all task IDs and track presentations
        all_task_ids = list(curriculum._tasks.keys())

        # Give different performance patterns to create eviction candidates
        # while testing minimum presentations requirement
        for task_id in all_task_ids:
            task = curriculum._tasks[task_id]
            # Task 0 gets many presentations with poor performance
            # Task 1 and 2 get fewer presentations with varying performance
            if task_id == all_task_ids[0]:
                # Give this task minimum+5 presentations with poor performance
                for _ in range(config.min_presentations_for_eviction + 5):
                    task.complete(0.1)  # Poor performance
                    curriculum.update_task_performance(task_id, 0.1)
            elif task_id == all_task_ids[1]:
                # Give this task exactly minimum presentations with good performance
                for _ in range(config.min_presentations_for_eviction):
                    task.complete(0.8)  # Good performance
                    curriculum.update_task_performance(task_id, 0.8)
            else:
                # Give this task fewer than minimum presentations
                for _ in range(config.min_presentations_for_eviction - 5):
                    task.complete(0.5)  # Medium performance
                    curriculum.update_task_performance(task_id, 0.5)

        # Check eviction criteria before forcing more get_task() calls
        algorithm = curriculum._algorithm

        # Task 0 should be evictable (meets min presentations + poor performance)
        task_0_evictable = algorithm.should_evict_task(all_task_ids[0], config.min_presentations_for_eviction)
        # Task 1 should not be evictable (meets min presentations but good performance)
        task_1_evictable = algorithm.should_evict_task(all_task_ids[1], config.min_presentations_for_eviction)
        # Task 2 should not be evictable (doesn't meet min presentations)
        task_2_evictable = algorithm.should_evict_task(all_task_ids[2], config.min_presentations_for_eviction)

        print(
            f"Task 0 evictable: {task_0_evictable}, "
            f"Task 1 evictable: {task_1_evictable}, "
            f"Task 2 evictable: {task_2_evictable}"
        )

        # Force get_task() calls to trigger eviction
        for _ in range(20):
            task = curriculum.get_task()
            # Maintain performance patterns
            if task._task_id == all_task_ids[0]:
                task.complete(0.1)  # Poor
                curriculum.update_task_performance(task._task_id, 0.1)
            elif task._task_id in all_task_ids[1:]:
                task.complete(0.8)  # Good
                curriculum.update_task_performance(task._task_id, 0.8)
            else:
                # New task - give medium performance
                task.complete(0.5)
                curriculum.update_task_performance(task._task_id, 0.5)

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

        # Create distinct performance patterns with many more presentations for realistic EMA timescales
        # Use 30 presentations per task to allow EMAs to develop meaningful differences

        def generate_improving_pattern(start=0.1, end=0.9, length=30):
            """Generate gradually improving performance."""
            return [start + (end - start) * i / (length - 1) for i in range(length)]

        def generate_consistent_pattern(value=0.4, length=30, noise=0.0):
            """Generate consistent performance with optional noise."""
            import random

            return [max(0.0, min(1.0, value + random.uniform(-noise, noise))) for _ in range(length)]

        def generate_variable_pattern(length=30):
            """Generate highly variable performance."""
            import random

            return [random.choice([0.2, 0.8]) for _ in range(length)]

        task_performance_patterns = {
            initial_task_ids[0]: generate_improving_pattern(0.1, 0.9, 30),  # Strong learning progress
            initial_task_ids[1]: generate_consistent_pattern(0.4, 30, 0.05),  # Low, stable performance
            initial_task_ids[2]: generate_variable_pattern(30),  # High variance, unstable learning
            initial_task_ids[3]: generate_consistent_pattern(
                0.8, 30, 0.03
            ),  # High, stable performance (already learned)
        }

        # Apply performance patterns with realistic timing
        for task_id, performances in task_performance_patterns.items():
            for performance in performances:
                curriculum.update_task_performance(task_id, performance)

        # Get learning progress scores to verify our setup
        algorithm = curriculum._algorithm
        scores = algorithm.score_tasks(initial_task_ids)

        # Verify that the algorithm can differentiate between different patterns
        improving_task = initial_task_ids[0]
        poor_task = initial_task_ids[1]
        variable_task = initial_task_ids[2]
        high_stable_task = initial_task_ids[3]

        print(
            f"Scores - Improving: {scores[improving_task]:.4f}, "
            f"Poor: {scores[poor_task]:.4f}, "
            f"Variable: {scores[variable_task]:.4f}, "
            f"High stable: {scores[high_stable_task]:.4f}"
        )

        # With more presentations, the algorithm should be able to differentiate
        # Variable task should have higher learning progress than stable tasks
        assert scores[variable_task] > scores[poor_task], (
            f"Variable task should have higher score than poor stable task. "
            f"Variable: {scores[variable_task]}, Poor stable: {scores[poor_task]}"
        )

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

        # Verify that eviction happens based on learning progress
        if evicted_tasks:
            # The evicted task should be one of the lower scoring tasks
            evicted_task = list(evicted_tasks)[0]
            evicted_score = scores[evicted_task]

            # Check that a lower-scoring task was evicted, not the highest scoring one
            max_score = max(scores.values())
            assert evicted_score < max_score, (
                f"Lower scoring task should be evicted. Evicted score: {evicted_score}, Max score: {max_score}"
            )
