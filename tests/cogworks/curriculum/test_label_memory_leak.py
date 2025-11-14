"""Test that label tracking doesn't leak memory over long training runs.

This test verifies the fix for the memory leak where _label_completion_counts
and _label_sampling_counts would grow unbounded with inactive labels.
"""

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig, CurriculumTask
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from mettagrid.config.mettagrid_config import MettaGridConfig


class TestLabelMemoryLeak:
    """Test that inactive label tracking is properly cleaned up."""

    def test_inactive_labels_are_cleaned_up_after_eviction(self):
        """Test that inactive labels are removed from tracking after exceeding retention limit."""
        # Create curriculum with reasonable limits for testing
        # Use defer_init=True to prevent auto-initialization of tasks
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=20,  # Enough to track all tasks we'll create
            max_inactive_labels_retained=5,  # Small limit for testing cleanup
            ema_timescale=0.1,
            exploration_bonus=0.1,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen,
            algorithm_config=algorithm_config,
            num_active_tasks=20,
            seed=42,
            defer_init=True,  # Don't auto-create tasks
        )
        curriculum = Curriculum(config)
        algorithm = curriculum._algorithm

        # Create 20 tasks with unique labels
        for i in range(20):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            task._label = f"label_{i}"
            algorithm.on_task_created(task)
            algorithm.update_task_performance(i, 0.5)

        # Simulate eviction of first 15 tasks (making their labels inactive)
        for i in range(15):
            algorithm.on_task_evicted(i)

        # Check that only the most recent 5 inactive labels are retained
        label_completion_counts = algorithm.task_tracker.get_label_completion_counts()
        total_labels = len(label_completion_counts)
        active_labels = len(algorithm._active_labels)
        inactive_labels_tracked = len(algorithm._inactive_labels_fifo)

        # After evicting 15 tasks, we should have:
        # - 5 active labels (tasks 15-19 still in pool)
        # - 5 inactive labels in FIFO (most recent evictions: labels 10-14)
        # - Completion counts only for the 5 remaining active tasks
        assert active_labels == 5, f"Expected 5 active labels, got {active_labels}"
        assert inactive_labels_tracked == 5, f"Expected 5 inactive labels in FIFO, got {inactive_labels_tracked}"
        assert total_labels == 5, f"Expected 5 active labels in completion counts, got {total_labels}"

    def test_label_reactivation_removes_from_inactive_queue(self):
        """Test that reactivating a label removes it from the inactive queue."""
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            max_inactive_labels_retained=10,
            ema_timescale=0.1,
            exploration_bonus=0.1,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen, algorithm_config=algorithm_config, num_active_tasks=10, seed=42, defer_init=True
        )
        curriculum = Curriculum(config)
        algorithm = curriculum._algorithm

        # Create and evict task with label_1
        task1 = CurriculumTask(task_id=1, env_cfg=MettaGridConfig(), slice_values={})
        task1._label = "label_1"
        algorithm.on_task_created(task1)
        algorithm.update_task_performance(1, 0.5)

        # Before evicting, verify label is tracked
        assert algorithm.task_tracker.get_task_label(1) == "label_1"

        algorithm.on_task_evicted(1)

        # Verify label_1 is in inactive queue
        assert "label_1" in algorithm._inactive_labels_fifo
        assert "label_1" not in algorithm._active_labels

        # Create new task with same label (reactivation)
        task2 = CurriculumTask(task_id=2, env_cfg=MettaGridConfig(), slice_values={})
        task2._label = "label_1"
        algorithm.on_task_created(task2)

        # Verify label_1 is removed from inactive queue and added back to active
        assert "label_1" not in algorithm._inactive_labels_fifo
        assert "label_1" in algorithm._active_labels

    def test_stats_preserved_for_retained_inactive_labels(self):
        """Test that stats are preserved for inactive labels within retention window."""
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            max_inactive_labels_retained=10,
            ema_timescale=0.1,
            exploration_bonus=0.1,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen, algorithm_config=algorithm_config, num_active_tasks=10, seed=42, defer_init=True
        )
        curriculum = Curriculum(config)
        algorithm = curriculum._algorithm

        # Create task and record some stats
        task = CurriculumTask(task_id=1, env_cfg=MettaGridConfig(), slice_values={})
        task._label = "test_label"
        algorithm.on_task_created(task)
        algorithm.update_task_performance(1, 0.8)
        algorithm.update_task_performance(1, 0.9)
        algorithm.on_task_sampled(1)
        algorithm.on_task_sampled(1)

        # Get sampling counts (completion counts will disappear after eviction)
        sampling_count_before = algorithm._label_sampling_counts.get("test_label", 0)

        # Evict the task
        algorithm.on_task_evicted(1)

        # After eviction, the task is marked inactive, so it won't appear in completion counts
        # But sampling counts (local state) should still be there
        label_completion_counts_after = algorithm.task_tracker.get_label_completion_counts()
        # Completion counts won't include inactive tasks
        assert "test_label" not in label_completion_counts_after
        # But sampling counts should be preserved locally
        assert algorithm._label_sampling_counts.get("test_label") == sampling_count_before

    def test_checkpoint_includes_inactive_labels_fifo(self):
        """Test that checkpointing preserves the inactive labels tracking."""
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            max_inactive_labels_retained=10,
            ema_timescale=0.1,
            exploration_bonus=0.1,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen, algorithm_config=algorithm_config, num_active_tasks=10, seed=42, defer_init=True
        )
        curriculum = Curriculum(config)
        algorithm = curriculum._algorithm

        # Create and evict some tasks
        for i in range(5):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            task._label = f"label_{i}"
            algorithm.on_task_created(task)
            algorithm.update_task_performance(i, 0.5)
            algorithm.on_task_evicted(i)

        # Get state
        state = algorithm.get_state()

        # Verify state includes label tracking
        assert "label_tracking" in state
        assert "inactive_labels_fifo" in state["label_tracking"]
        assert len(state["label_tracking"]["inactive_labels_fifo"]) == 5

        # Create new algorithm and load state
        config2 = CurriculumConfig(
            task_generator=task_gen, algorithm_config=algorithm_config, num_active_tasks=10, seed=42, defer_init=True
        )
        curriculum2 = Curriculum(config2)
        algorithm2 = curriculum2._algorithm
        algorithm2.load_state(state)

        # Verify restored state
        assert algorithm2._inactive_labels_fifo == algorithm._inactive_labels_fifo
        # Completion counts are now in TaskTracker, compare those instead
        label_counts1 = algorithm.task_tracker.get_label_completion_counts()
        label_counts2 = algorithm2.task_tracker.get_label_completion_counts()
        assert label_counts2 == label_counts1
        assert algorithm2._label_sampling_counts == algorithm._label_sampling_counts
