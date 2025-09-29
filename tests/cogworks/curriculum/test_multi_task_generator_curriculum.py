"""Tests for multi-task generator curriculum functionality."""

import pytest

from metta.cogworks.curriculum import (
    CurriculumConfig,
    SingleTaskGenerator,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from mettagrid.config.mettagrid_config import MettaGridConfig


class TestMultiTaskGeneratorCurriculumConfig:
    """Test configuration validation for multi-task generator curriculum."""

    @pytest.fixture
    def single_task_config(self):
        """Create a single task generator config."""
        return SingleTaskGenerator.Config(env=MettaGridConfig(label="task_1"))

    @pytest.fixture
    def multiple_task_configs(self):
        """Create multiple task generator configs."""
        return [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="task_1")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="task_2")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="task_3")),
        ]

    def test_single_task_generator_backward_compatibility(self, single_task_config):
        """Test that single task generator mode still works (backward compatibility)."""
        config = CurriculumConfig(
            task_generator=single_task_config,
            num_active_tasks=10,
        )

        assert config.task_generator is single_task_config
        assert config.task_generators is None
        assert config.min_generator_proportion == 0.1  # Default value

        # Should be able to create curriculum
        curriculum = config.make()
        assert hasattr(curriculum, "_is_multi_generator")
        assert not curriculum._is_multi_generator

    def test_multiple_task_generators_config(self, multiple_task_configs):
        """Test that multiple task generator mode works correctly."""
        config = CurriculumConfig(
            task_generators=multiple_task_configs,
            num_active_tasks=15,
            min_generator_proportion=0.2,
        )

        assert config.task_generator is None
        assert config.task_generators == multiple_task_configs
        assert config.min_generator_proportion == 0.2

        # Should be able to create curriculum
        curriculum = config.make()
        assert hasattr(curriculum, "_is_multi_generator")
        assert curriculum._is_multi_generator
        assert len(curriculum._task_generators) == 3

    def test_config_validation_both_generators_provided(self, single_task_config, multiple_task_configs):
        """Test that providing both single and multiple generators raises error."""
        with pytest.raises(ValueError, match="Cannot provide both task_generator and task_generators"):
            CurriculumConfig(
                task_generator=single_task_config,
                task_generators=multiple_task_configs,
                num_active_tasks=10,
            )

    def test_config_validation_no_generators_provided(self):
        """Test that providing no generators raises error."""
        with pytest.raises(ValueError, match="Must provide either task_generator or task_generators"):
            CurriculumConfig(num_active_tasks=10)

    def test_config_validation_empty_task_generators(self):
        """Test that empty task_generators list raises error."""
        with pytest.raises(ValueError, match="Must provide at least one task generator"):
            CurriculumConfig(
                task_generators=[],
                num_active_tasks=10,
            )

    def test_config_validation_min_proportion_too_high(self, multiple_task_configs):
        """Test that min_generator_proportion constraint is validated."""
        with pytest.raises(ValueError, match="min_generator_proportion .* cannot exceed 1.0"):
            CurriculumConfig(
                task_generators=multiple_task_configs,  # 3 generators
                num_active_tasks=10,
                min_generator_proportion=0.4,  # 0.4 * 3 = 1.2 > 1.0
            )


class TestMultiTaskGeneratorCurriculumBasics:
    """Test basic functionality of multi-task generator curriculum."""

    @pytest.fixture
    def multi_generator_curriculum(self):
        """Create a curriculum with multiple generators."""
        generators = [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="generator_0")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="generator_1")),
        ]

        config = CurriculumConfig(
            task_generators=generators,
            num_active_tasks=10,
            min_generator_proportion=0.3,
            algorithm_config=LearningProgressConfig(num_active_tasks=10),
        )

        return config.make()

    def test_multi_generator_initialization(self, multi_generator_curriculum):
        """Test that multi-generator curriculum initializes correctly."""
        curriculum = multi_generator_curriculum

        assert curriculum._is_multi_generator
        assert len(curriculum._task_generators) == 2
        assert len(curriculum._generator_task_counts) == 2
        assert len(curriculum._task_to_generator) >= 0
        assert curriculum._config.min_generator_proportion == 0.3

    def test_task_generation_uses_both_generators(self, multi_generator_curriculum):
        """Test that tasks are generated from both generators."""
        curriculum = multi_generator_curriculum

        # Generate several tasks and track which generators are used
        generator_usage = {0: 0, 1: 0}
        generated_tasks = []

        for _ in range(20):
            task = curriculum.get_task()
            generated_tasks.append(task)

            # Track which generator was used based on label
            if "generator_0" in task.get_env_cfg().label:
                generator_usage[0] += 1
            elif "generator_1" in task.get_env_cfg().label:
                generator_usage[1] += 1

        # Both generators should have been used
        assert generator_usage[0] > 0, "Generator 0 should have been used"
        assert generator_usage[1] > 0, "Generator 1 should have been used"

        # Total should match number of tasks generated
        assert generator_usage[0] + generator_usage[1] == 20

    def test_task_to_generator_mapping(self, multi_generator_curriculum):
        """Test that task-to-generator mapping is maintained correctly."""
        curriculum = multi_generator_curriculum

        # Generate tasks and verify mapping
        tasks = []
        for _ in range(10):
            task = curriculum.get_task()
            tasks.append(task)

            # Check that task is mapped to a generator
            assert task._task_id in curriculum._task_to_generator
            generator_idx = curriculum._task_to_generator[task._task_id]
            assert 0 <= generator_idx < len(curriculum._task_generators)

    def test_generator_task_counts(self, multi_generator_curriculum):
        """Test that generator task counts are maintained correctly."""
        curriculum = multi_generator_curriculum

        # The curriculum initializes at capacity, so counts should already reflect that
        initial_counts = curriculum._generator_task_counts.copy()
        total_initial_tasks = sum(initial_counts)

        # Total initial tasks should equal the pool size
        assert total_initial_tasks == curriculum._config.num_active_tasks

        # Both generators should have been used during initialization
        assert all(count > 0 for count in initial_counts), "All generators should have created tasks"

        # Generate some tasks (these will be existing tasks since we're at capacity)
        for _ in range(5):
            task = curriculum.get_task()
            # Verify the task came from one of our generators
            assert task._task_id in curriculum._task_to_generator

        # Counts should not have changed since we're just returning existing tasks
        final_counts = curriculum._generator_task_counts
        assert final_counts == initial_counts, "Task counts should not change when returning existing tasks"


class TestMultiTaskGeneratorProportionalSampling:
    """Test proportional sampling based on performance scores."""

    @pytest.fixture
    def test_curriculum_for_scoring(self):
        """Create a curriculum specifically for testing scoring behavior."""
        generators = [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="low_performer")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="high_performer")),
        ]

        config = CurriculumConfig(
            task_generators=generators,
            num_active_tasks=20,
            min_generator_proportion=0.1,  # Low minimum to test scoring effect
            algorithm_config=LearningProgressConfig(num_active_tasks=20),
        )

        return config.make()

    def test_minimum_proportion_constraint(self):
        """Test that minimum proportion constraint is enforced."""
        generators = [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="generator_0")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="generator_1")),
        ]

        config = CurriculumConfig(
            task_generators=generators,
            num_active_tasks=10,
            min_generator_proportion=0.4,  # 40% minimum
        )

        curriculum = config.make()

        # Force generation of many tasks to see proportions
        generator_counts = [0, 0]
        for _ in range(100):
            task = curriculum.get_task()

            # Update task performance (neutral scores)
            curriculum.update_task_performance(task._task_id, 0.5)
            task.complete(0.5)

            # Track which generator was used
            if "generator_0" in task.get_env_cfg().label:
                generator_counts[0] += 1
            else:
                generator_counts[1] += 1

        # Each generator should have at least 30% (allowing some randomness around 40%)
        total_tasks = sum(generator_counts)
        for count in generator_counts:
            proportion = count / total_tasks
            assert proportion >= 0.3, f"Generator proportion {proportion} below expected minimum"

    def test_performance_based_sampling(self, test_curriculum_for_scoring):
        """Test that better performing generators are sampled more frequently."""
        curriculum = test_curriculum_for_scoring

        # Get all existing tasks from the pool (curriculum initializes at capacity)
        all_tasks = list(curriculum._tasks.values())

        # Simulate different performance patterns
        low_performer_tasks = []
        high_performer_tasks = []

        for task in all_tasks:
            if "low_performer" in task.get_env_cfg().label:
                low_performer_tasks.append(task)
            else:
                high_performer_tasks.append(task)

        # Give enough presentations and low performance to low_performer tasks
        # This should make them eligible for eviction
        for task in low_performer_tasks:
            for i in range(8):  # More than min_presentations (5)
                score = 0.1 + (i % 3) * 0.05  # Low, varying scores
                curriculum.update_task_performance(task._task_id, score)
                task.complete(score)

        # Give enough presentations and high performance to high_performer tasks
        for task in high_performer_tasks:
            for i in range(8):  # More than min_presentations (5)
                score = 0.8 + (i % 3) * 0.05  # High, varying scores
                curriculum.update_task_performance(task._task_id, score)
                task.complete(score)

        # Now when we call get_task(), it should trigger evictions of low performers
        # and create new tasks using the generator selection logic
        generator_counts = {"low_performer": 0, "high_performer": 0}

        # Force many task requests to trigger evictions and new task creation
        for _ in range(100):  # More iterations to see the effect
            task = curriculum.get_task()

            # Schedule the task multiple times to increase presentations
            task._num_scheduled += 1

            if "low_performer" in task.get_env_cfg().label:
                generator_counts["low_performer"] += 1
            else:
                generator_counts["high_performer"] += 1

            # Give the task some performance to continue the pattern
            if "low_performer" in task.get_env_cfg().label:
                score = 0.15  # Keep low
            else:
                score = 0.85  # Keep high

            curriculum.update_task_performance(task._task_id, score)
            task.complete(score)

        # Due to eviction and replacement, high performer should be favored
        # when creating new tasks to replace evicted low performers
        high_performer_ratio = generator_counts["high_performer"] / 100

        # High performer should get more than 50% due to better mean scores
        # Note: minimum proportion constraint (0.1) still applies
        assert high_performer_ratio > 0.55, (
            f"High performer ratio {high_performer_ratio} should be > 0.55 (got counts: {generator_counts})"
        )

    def test_stats_include_generator_metrics(self, test_curriculum_for_scoring):
        """Test that statistics include per-generator metrics."""
        curriculum = test_curriculum_for_scoring

        # Generate some tasks and give them scores
        for _ in range(10):
            task = curriculum.get_task()
            curriculum.update_task_performance(task._task_id, 0.5)
            task.complete(0.5)

        stats = curriculum.get_base_stats()

        # Should have multi-generator stats
        assert "num_generators" in stats
        assert stats["num_generators"] == 2.0

        # Should have per-generator task counts (cumulative)
        assert "generator_0_task_count" in stats
        assert "generator_1_task_count" in stats

        # Should have per-generator pool counts (current active pool)
        assert "generator_0_pool_count" in stats
        assert "generator_1_pool_count" in stats

        # Should have per-generator pool proportions
        assert "generator_0_pool_proportion" in stats
        assert "generator_1_pool_proportion" in stats

        # Should have per-generator mean scores
        assert "generator_0_mean_score" in stats
        assert "generator_1_mean_score" in stats

        # Pool counts should add up to total active tasks
        total_pool_tasks = stats["generator_0_pool_count"] + stats["generator_1_pool_count"]
        assert total_pool_tasks == stats["num_active_tasks"]

        # Pool proportions should add up to 1.0
        total_proportion = stats["generator_0_pool_proportion"] + stats["generator_1_pool_proportion"]
        assert abs(total_proportion - 1.0) < 1e-6, f"Pool proportions should sum to 1.0, got {total_proportion}"


class TestMultiTaskGeneratorStateManagement:
    """Test state saving and loading for multi-task generator curriculum."""

    @pytest.fixture
    def stateful_curriculum(self):
        """Create a curriculum for state management testing."""
        generators = [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="state_gen_0")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="state_gen_1")),
        ]

        config = CurriculumConfig(
            task_generators=generators,
            num_active_tasks=8,
            min_generator_proportion=0.2,
        )

        return config.make()

    def test_state_save_and_load(self, stateful_curriculum):
        """Test that curriculum state can be saved and loaded correctly."""
        curriculum = stateful_curriculum

        # Generate some tasks and give them scores
        tasks = []
        for _ in range(5):
            task = curriculum.get_task()
            tasks.append(task)
            curriculum.update_task_performance(task._task_id, 0.7)
            task.complete(0.7)

        # Save state
        state = curriculum.get_state()

        # Verify state contains multi-generator information
        assert "is_multi_generator" in state
        assert state["is_multi_generator"] is True
        assert "task_to_generator" in state
        assert "generator_task_counts" in state

        # Create new curriculum and load state
        new_curriculum = stateful_curriculum._config.make()
        new_curriculum.load_state(state)

        # Verify state was loaded correctly
        assert new_curriculum._is_multi_generator
        assert new_curriculum._task_to_generator == curriculum._task_to_generator
        assert new_curriculum._generator_task_counts == curriculum._generator_task_counts
        assert len(new_curriculum._tasks) == len(curriculum._tasks)

        # Tasks should have same properties
        for task_id in curriculum._tasks:
            assert task_id in new_curriculum._tasks
            original_task = curriculum._tasks[task_id]
            loaded_task = new_curriculum._tasks[task_id]
            assert original_task._mean_score == loaded_task._mean_score
            assert original_task._num_completions == loaded_task._num_completions


class TestMultiTaskGeneratorEviction:
    """Test task eviction behavior in multi-task generator curriculum."""

    def test_eviction_updates_generator_counts(self):
        """Test that task eviction correctly updates generator tracking."""
        generators = [
            SingleTaskGenerator.Config(env=MettaGridConfig(label="evict_gen_0")),
            SingleTaskGenerator.Config(env=MettaGridConfig(label="evict_gen_1")),
        ]

        config = CurriculumConfig(
            task_generators=generators,
            num_active_tasks=6,  # Small pool to force evictions
            min_generator_proportion=0.2,
            algorithm_config=LearningProgressConfig(num_active_tasks=6),
        )

        curriculum = config.make()

        # Get all initial tasks and give them enough presentations
        all_tasks = list(curriculum._tasks.values())

        # Give some tasks high scores, some low scores, and enough presentations
        for i, task in enumerate(all_tasks):
            # Give enough presentations (more than min_presentations of 5)
            for j in range(8):
                if i < len(all_tasks) // 2:
                    # First half gets low scores (should be evicted)
                    score = 0.1 + (j % 3) * 0.02  # Low scores with variation
                else:
                    # Second half gets high scores (should not be evicted)
                    score = 0.8 + (j % 3) * 0.02  # High scores with variation

                curriculum.update_task_performance(task._task_id, score)
                task.complete(score)

        # Record initial state
        initial_evicted_count = curriculum._num_evicted

        # Now force many task requests to trigger evictions
        # The low-scoring tasks should become eligible for eviction
        for _ in range(20):
            task = curriculum.get_task()

            # Continue giving performance scores
            if any(low_task._task_id == task._task_id for low_task in all_tasks[: len(all_tasks) // 2]):
                score = 0.1  # Keep low performers low
            else:
                score = 0.8  # Keep high performers high

            curriculum.update_task_performance(task._task_id, score)
            task.complete(score)

        # Pool should still be at capacity
        assert len(curriculum._tasks) == config.num_active_tasks

        # Generator counts should still add up to total active tasks
        final_counts = curriculum._generator_task_counts
        assert sum(final_counts) == len(curriculum._tasks)

        # Should have had some evictions (low scoring tasks should be evicted)
        final_evicted_count = curriculum._num_evicted
        assert final_evicted_count > initial_evicted_count, (
            f"Expected evictions, but evicted count didn't increase: {initial_evicted_count} -> {final_evicted_count}"
        )


class TestBackwardCompatibility:
    """Test that single task generator mode is unaffected by multi-generator changes."""

    def test_single_generator_mode_unchanged(self):
        """Test that single generator mode works exactly as before."""
        config = CurriculumConfig(
            task_generator=SingleTaskGenerator.Config(env=MettaGridConfig(label="single_task")),
            num_active_tasks=5,
        )

        curriculum = config.make()

        # Should be in single generator mode
        assert not curriculum._is_multi_generator
        assert curriculum._task_generator is not None
        assert curriculum._task_generators is None

        # Should not have multi-generator attributes
        assert not hasattr(curriculum, "_task_to_generator")
        assert not hasattr(curriculum, "_generator_task_counts")

        # Should generate tasks normally
        task = curriculum.get_task()
        assert "single_task" in task.get_env_cfg().label

        # Stats should not include multi-generator metrics
        stats = curriculum.get_base_stats()
        assert "num_generators" not in stats
        assert "generator_0_task_count" not in stats

    def test_single_generator_state_management(self):
        """Test that state management works for single generator mode."""
        config = CurriculumConfig(
            task_generator=SingleTaskGenerator.Config(env=MettaGridConfig(label="state_single")),
            num_active_tasks=3,
        )

        curriculum = config.make()

        # Generate and score a task
        task = curriculum.get_task()
        curriculum.update_task_performance(task._task_id, 0.8)
        task.complete(0.8)

        # Save and load state
        state = curriculum.get_state()
        assert state["is_multi_generator"] is False
        assert "task_to_generator" not in state
        assert "generator_task_counts" not in state

        # Load into new curriculum
        new_curriculum = config.make()
        new_curriculum.load_state(state)

        # Should work correctly
        assert not new_curriculum._is_multi_generator
        assert len(new_curriculum._tasks) == len(curriculum._tasks)
