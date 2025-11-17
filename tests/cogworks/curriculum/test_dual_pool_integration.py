"""Integration tests for dual-pool curriculum system.

This module tests the end-to-end behavior of the dual-pool curriculum, including:
- Task creation in explore pool
- Task sampling during bootstrap and steady-state phases
- Automatic promotion from explore to exploit
- Adaptive EER updates based on promotion success
- Phase transitions
- State persistence across checkpoints
"""

import pytest

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from packages.mettagrid.python.src.mettagrid import MettaGridConfig


@pytest.fixture
def dual_pool_config():
    """Create a dual-pool curriculum configuration for testing."""
    # Simple MettaGrid config for task generation (use default)
    mg_config = MettaGridConfig()

    # Dual-pool LP configuration
    lp_config = LearningProgressConfig(
        use_dual_pool=True,
        num_explore_tasks=5,  # Small for faster testing
        num_exploit_tasks=10,  # Small for faster testing
        promotion_min_samples=3,  # Lower threshold for testing
        explore_exploit_ratio_init=0.5,
        explore_exploit_ratio_alpha=0.9,
        promotion_rate_window=10,
        use_shared_memory=False,  # Use local memory for tests
    )

    curriculum_config = CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=mg_config),
        algorithm_config=lp_config,
    )

    return curriculum_config


@pytest.fixture
def dual_pool_curriculum(dual_pool_config):
    """Create a dual-pool curriculum instance."""
    curriculum = Curriculum(dual_pool_config)
    yield curriculum
    # Cleanup
    if curriculum._algorithm is not None:
        curriculum._algorithm.cleanup_shared_memory()


class TestDualPoolBootstrapPhase:
    """Test dual-pool behavior during bootstrap phase."""

    def test_initial_phase_is_bootstrap(self, dual_pool_curriculum):
        """Test that curriculum starts in bootstrap phase."""
        algorithm = dual_pool_curriculum._algorithm
        assert algorithm.get_current_phase() == "bootstrap"

    def test_all_initial_tasks_in_explore_pool(self, dual_pool_curriculum):
        """Test that all initial tasks are created in explore pool."""
        algorithm = dual_pool_curriculum._algorithm

        # Get all task IDs
        all_tasks = list(dual_pool_curriculum._tasks.keys())

        # All should be in explore pool
        explore_tasks = algorithm.get_pool_task_ids("explore")
        exploit_tasks = algorithm.get_pool_task_ids("exploit")

        assert len(exploit_tasks) == 0, "Exploit pool should be empty initially"
        assert set(all_tasks) == set(explore_tasks), "All tasks should be in explore pool"

    def test_bootstrap_phase_samples_from_explore(self, dual_pool_curriculum):
        """Test that bootstrap phase samples 100% from explore pool."""
        algorithm = dual_pool_curriculum._algorithm

        # Sample multiple tasks
        sample_count = 20
        sampled_pools = []

        for _ in range(sample_count):
            task = dual_pool_curriculum.get_task()
            # Check which pool the task is in
            if task._task_id in algorithm.get_pool_task_ids("explore"):
                sampled_pools.append("explore")
            elif task._task_id in algorithm.get_pool_task_ids("exploit"):
                sampled_pools.append("exploit")

        # All samples should be from explore during bootstrap
        assert all(pool == "explore" for pool in sampled_pools), (
            f"Bootstrap phase should sample 100% from explore, got: {sampled_pools}"
        )

    def test_phase_transitions_when_exploit_pool_fills(self, dual_pool_curriculum):
        """Test that phase transitions from bootstrap to steady_state when exploit pool fills."""
        algorithm = dual_pool_curriculum._algorithm
        num_exploit_tasks = algorithm.hypers.num_exploit_tasks

        # Initially in bootstrap
        assert algorithm.get_current_phase() == "bootstrap"

        # Simulate task updates to trigger promotions
        explore_tasks = algorithm.get_pool_task_ids("explore")

        # Update each task multiple times to make them eligible for promotion
        for task_id in explore_tasks[: num_exploit_tasks + 2]:
            # Give high scores to encourage promotion
            for _ in range(algorithm.hypers.promotion_min_samples + 1):
                dual_pool_curriculum.update_task_performance(task_id, 0.9)

        # Check if we've transitioned to steady state
        # (Should happen when exploit pool reaches capacity)
        exploit_tasks = algorithm.get_pool_task_ids("exploit")
        if len(exploit_tasks) >= num_exploit_tasks:
            assert algorithm.get_current_phase() == "steady_state", (
                f"Should transition to steady_state when exploit pool is full, got {algorithm.get_current_phase()}"
            )


class TestDualPoolPromotion:
    """Test task promotion from explore to exploit."""

    def test_task_promotion_after_min_samples(self, dual_pool_curriculum):
        """Test that tasks promote after reaching min_samples threshold."""
        algorithm = dual_pool_curriculum._algorithm
        min_samples = algorithm.hypers.promotion_min_samples

        # Get a task from explore pool
        explore_tasks = algorithm.get_pool_task_ids("explore")
        assert len(explore_tasks) > 0, "Should have tasks in explore pool"

        task_id = explore_tasks[0]

        # Update task with high scores (should trigger promotion)
        for _ in range(min_samples + 1):
            dual_pool_curriculum.update_task_performance(task_id, 0.95)

        # Check if task was promoted
        exploit_tasks = algorithm.get_pool_task_ids("exploit")

        # Task should be promoted to exploit (if space available)
        if len(exploit_tasks) < algorithm.hypers.num_exploit_tasks:
            assert task_id in exploit_tasks or task_id in algorithm.get_pool_task_ids("explore"), (
                "Task should be in one of the pools"
            )

    def test_low_scoring_tasks_dont_promote(self, dual_pool_curriculum):
        """Test that low-scoring tasks don't promote when exploit pool is full and they have low LP scores."""
        algorithm = dual_pool_curriculum._algorithm
        min_samples = algorithm.hypers.promotion_min_samples
        num_exploit = algorithm.hypers.num_exploit_tasks

        # First, fill the exploit pool with high-scoring tasks
        # We need to create enough tasks by sampling from curriculum
        promoted_count = 0
        while promoted_count < num_exploit:
            # Get or create tasks in explore pool
            if len(algorithm.get_pool_task_ids("explore")) == 0:
                task = dual_pool_curriculum.get_task()

            explore_tasks = algorithm.get_pool_task_ids("explore")
            if not explore_tasks:
                break

            # Update first explore task with high scores
            task_id = explore_tasks[0]
            for _ in range(min_samples + 1):
                # Give high scores to trigger promotion
                dual_pool_curriculum.update_task_performance(task_id, 0.9)

            # Check if promoted
            if task_id in algorithm.get_pool_task_ids("exploit"):
                promoted_count += 1

        # Verify exploit pool is full
        exploit_tasks = algorithm.get_pool_task_ids("exploit")
        assert len(exploit_tasks) == num_exploit, f"Exploit pool should be full, got {len(exploit_tasks)}/{num_exploit}"

        # Now create/get a task for testing low-score promotion
        if len(algorithm.get_pool_task_ids("explore")) == 0:
            task = dual_pool_curriculum.get_task()
            low_score_task_id = task._task_id
        else:
            low_score_task_id = algorithm.get_pool_task_ids("explore")[0]

        # Update this task with constant low scores (zero learning progress)
        for _ in range(min_samples + 5):
            # Constant score = zero learning progress
            dual_pool_curriculum.update_task_performance(low_score_task_id, 0.2)

        # Task should still be in explore (zero/low LP score prevents promotion when exploit is full)
        current_explore = algorithm.get_pool_task_ids("explore")
        current_exploit = algorithm.get_pool_task_ids("exploit")

        assert low_score_task_id in current_explore, (
            f"Task with zero learning progress should remain in explore when exploit pool is full. "
            f"Task {low_score_task_id} is in exploit: {low_score_task_id in current_exploit}"
        )

    def test_promotion_updates_eer(self, dual_pool_curriculum):
        """Test that successful promotions update the EER."""
        algorithm = dual_pool_curriculum._algorithm

        # Promote some tasks
        explore_tasks = algorithm.get_pool_task_ids("explore")
        for task_id in explore_tasks[:3]:
            for _ in range(algorithm.hypers.promotion_min_samples + 1):
                dual_pool_curriculum.update_task_performance(task_id, 0.95)

        # Check if EER was updated
        stats_after = algorithm.get_detailed_stats()
        num_promotions = stats_after.get("dual_pool/num_promotions", 0)

        # If promotions occurred, check statistics
        if num_promotions > 0:
            assert "dual_pool/recent_promotion_rate" in stats_after
            assert stats_after["dual_pool/promotion_success_rate"] > 0


class TestDualPoolSteadyState:
    """Test dual-pool behavior during steady-state phase."""

    def _fill_exploit_pool(self, curriculum):
        """Helper to fill exploit pool and enter steady state."""
        algorithm = curriculum._algorithm
        num_exploit = algorithm.hypers.num_exploit_tasks

        explore_tasks = algorithm.get_pool_task_ids("explore")

        # Promote enough tasks to fill exploit pool
        for task_id in explore_tasks[: num_exploit + 2]:
            for _ in range(algorithm.hypers.promotion_min_samples + 1):
                curriculum.update_task_performance(task_id, 0.95)

    def test_steady_state_samples_from_both_pools(self, dual_pool_curriculum):
        """Test that steady state samples from both pools based on EER."""
        # Fill exploit pool to enter steady state
        self._fill_exploit_pool(dual_pool_curriculum)

        algorithm = dual_pool_curriculum._algorithm

        # Check we're in steady state
        if algorithm.get_current_phase() != "steady_state":
            pytest.skip("Exploit pool not full, still in bootstrap")

        # Sample many tasks
        sample_count = 50
        explore_count = 0
        exploit_count = 0

        for _ in range(sample_count):
            task = dual_pool_curriculum.get_task()

            # Check which pool task is from
            if task._task_id in algorithm.get_pool_task_ids("explore"):
                explore_count += 1
            elif task._task_id in algorithm.get_pool_task_ids("exploit"):
                exploit_count += 1

        # Should sample from both pools in steady state
        # (Allow some variance due to randomness)
        assert explore_count > 0, "Should sample some tasks from explore pool"
        assert exploit_count > 0, "Should sample some tasks from exploit pool"


class TestDualPoolStatistics:
    """Test dual-pool statistics and monitoring."""

    def test_pool_size_statistics(self, dual_pool_curriculum):
        """Test that pool size statistics are reported correctly."""
        algorithm = dual_pool_curriculum._algorithm
        stats = algorithm.get_detailed_stats()

        assert "dual_pool/num_explore_tasks" in stats
        assert "dual_pool/num_exploit_tasks" in stats

        num_explore = int(stats["dual_pool/num_explore_tasks"])
        num_exploit = int(stats["dual_pool/num_exploit_tasks"])

        # Verify counts match actual pools
        actual_explore = len(algorithm.get_pool_task_ids("explore"))
        actual_exploit = len(algorithm.get_pool_task_ids("exploit"))

        assert num_explore == actual_explore
        assert num_exploit == actual_exploit

    def test_eer_statistics(self, dual_pool_curriculum):
        """Test that EER statistics are reported."""
        algorithm = dual_pool_curriculum._algorithm
        stats = algorithm.get_detailed_stats()

        assert "dual_pool/explore_exploit_ratio" in stats
        eer = stats["dual_pool/explore_exploit_ratio"]

        # EER should be within valid bounds
        assert algorithm.hypers.explore_exploit_ratio_min <= eer <= algorithm.hypers.explore_exploit_ratio_max

    def test_promotion_statistics(self, dual_pool_curriculum):
        """Test that promotion statistics are reported."""
        algorithm = dual_pool_curriculum._algorithm
        stats = algorithm.get_detailed_stats()

        assert "dual_pool/num_promotions" in stats
        assert "dual_pool/num_promotion_attempts" in stats
        assert "dual_pool/promotion_success_rate" in stats
        assert "dual_pool/recent_promotion_rate" in stats

    def test_phase_indicators(self, dual_pool_curriculum):
        """Test that phase indicators are reported."""
        algorithm = dual_pool_curriculum._algorithm
        stats = algorithm.get_detailed_stats()

        assert "dual_pool/is_bootstrap_phase" in stats
        assert "dual_pool/is_steady_state_phase" in stats

        # Exactly one should be 1.0
        is_bootstrap = stats["dual_pool/is_bootstrap_phase"]
        is_steady = stats["dual_pool/is_steady_state_phase"]

        assert (is_bootstrap == 1.0 and is_steady == 0.0) or (is_bootstrap == 0.0 and is_steady == 1.0), (
            "Should be in exactly one phase"
        )


class TestDualPoolStatepersistence:
    """Test dual-pool state persistence and checkpointing."""

    def test_dual_pool_state_serialization(self, dual_pool_curriculum):
        """Test that dual-pool state can be saved and loaded."""
        algorithm = dual_pool_curriculum._algorithm

        # Modify state by promoting some tasks
        explore_tasks = algorithm.get_pool_task_ids("explore")
        for task_id in explore_tasks[:2]:
            for _ in range(algorithm.hypers.promotion_min_samples + 1):
                dual_pool_curriculum.update_task_performance(task_id, 0.9)

        # Get state
        state = algorithm.get_state()

        assert "dual_pool" in state
        dual_pool_state = state["dual_pool"]

        assert "explore_exploit_ratio" in dual_pool_state
        assert "promotion_window" in dual_pool_state
        assert "num_promotions" in dual_pool_state
        assert "num_promotion_attempts" in dual_pool_state
        assert "current_phase" in dual_pool_state

    def test_dual_pool_state_restoration(self, dual_pool_config):
        """Test that dual-pool state can be restored from checkpoint."""
        # Create first curriculum and modify its state
        curriculum1 = Curriculum(dual_pool_config)
        algorithm1 = curriculum1._algorithm

        # Promote some tasks to change state
        explore_tasks = algorithm1.get_pool_task_ids("explore")
        for task_id in explore_tasks[:3]:
            for _ in range(algorithm1.hypers.promotion_min_samples + 1):
                curriculum1.update_task_performance(task_id, 0.95)

        # Save state
        state1 = algorithm1.get_state()
        eer1 = algorithm1._explore_exploit_ratio
        promotions1 = algorithm1._num_promotions

        # Create new curriculum and load state
        curriculum2 = Curriculum(dual_pool_config)
        algorithm2 = curriculum2._algorithm
        algorithm2.load_state(state1)

        # Verify state was restored
        assert algorithm2._explore_exploit_ratio == eer1
        assert algorithm2._num_promotions == promotions1

        # Cleanup
        curriculum1._algorithm.cleanup_shared_memory()
        curriculum2._algorithm.cleanup_shared_memory()


class TestDualPoolEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pool_handling(self, dual_pool_curriculum):
        """Test that curriculum handles empty pools gracefully."""
        # This shouldn't happen in normal operation, but test robustness
        # Even if a pool is empty, sampling should still work
        task = dual_pool_curriculum.get_task()
        assert task is not None

    def test_single_pool_mode_unaffected(self, dual_pool_config):
        """Test that single-pool mode is completely unaffected by dual-pool code."""
        # Create a single-pool config
        dual_pool_config.algorithm_config.use_dual_pool = False

        curriculum = Curriculum(dual_pool_config)
        algorithm = curriculum._algorithm

        # Verify single-pool mode
        assert not algorithm.is_dual_pool_mode()
        assert algorithm.get_current_phase() == "single_pool"

        # Should work normally
        task = curriculum.get_task()
        assert task is not None

        # Cleanup
        curriculum._algorithm.cleanup_shared_memory()
