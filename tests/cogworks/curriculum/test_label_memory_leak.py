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
        # Create curriculum with small retention limit for testing
        # Use defer_init=True to prevent auto-initialization of tasks
        task_gen = SingleTaskGenerator.Config(env=MettaGridConfig())
        algorithm_config = LearningProgressConfig(
            num_active_tasks=10,
            max_inactive_labels_retained=5,  # Small limit for testing
            ema_timescale=0.1,
            exploration_bonus=0.1,
            use_shared_memory=False,
        )
        config = CurriculumConfig(
            task_generator=task_gen,
            algorithm_config=algorithm_config,
            num_active_tasks=10,
            seed=42,
            defer_init=True,  # Don't auto-create tasks
        )
        curriculum = Curriculum(config)
        algorithm = curriculum._algorithm

        # Create tasks with unique labels
        for i in range(20):
            task = CurriculumTask(task_id=i, env_cfg=MettaGridConfig(), slice_values={})
            task._label = f"label_{i}"
            algorithm.on_task_created(task)
            algorithm.update_task_performance(i, 0.5)

        # Simulate eviction of first 15 tasks (making their labels inactive)
        for i in range(15):
            algorithm.on_task_evicted(i)

        # Check that only the most recent 5 inactive labels are retained
        # (Plus any still-active labels from tasks 15-19)
        total_labels = len(algorithm._label_completion_counts)
        active_labels = len(algorithm._active_labels)
        inactive_labels_tracked = len(algorithm._inactive_labels_fifo)

        # Should have: 5 active labels (tasks 15-19) + 5 retained inactive labels = 10 total
        assert total_labels == 10, f"Expected 10 labels tracked, got {total_labels}"
        assert active_labels == 5, f"Expected 5 active labels, got {active_labels}"
        assert inactive_labels_tracked == 5, f"Expected 5 inactive labels in FIFO, got {inactive_labels_tracked}"

        # Verify old labels were cleaned up
        assert "label_0" not in algorithm._label_completion_counts
        assert "label_1" not in algorithm._label_completion_counts

        # Verify recent inactive labels are still there
        assert "label_10" in algorithm._label_completion_counts
        assert "label_14" in algorithm._label_completion_counts

        # Verify active labels are still there
        assert "label_15" in algorithm._label_completion_counts
        assert "label_19" in algorithm._label_completion_counts

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

        completion_count_before = algorithm._label_completion_counts.get("test_label", 0)
        sampling_count_before = algorithm._label_sampling_counts.get("test_label", 0)

        # Evict the task
        algorithm.on_task_evicted(1)

        # Stats should still be there (within retention window)
        assert algorithm._label_completion_counts.get("test_label") == completion_count_before
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
        assert algorithm2._label_completion_counts == algorithm._label_completion_counts
        assert algorithm2._label_sampling_counts == algorithm._label_sampling_counts
