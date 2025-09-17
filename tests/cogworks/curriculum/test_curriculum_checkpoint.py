"""Tests for curriculum checkpointing functionality."""

import tempfile
from pathlib import Path

from metta.cogworks.curriculum import Curriculum, env_curriculum
from metta.mettagrid.builder.envs import make_arena
from metta.rl.checkpoint_manager import CheckpointManager


class TestCurriculumCheckpointing:
    def test_curriculum_checkpoint_roundtrip(self):
        """Test that curriculum state can be saved and loaded correctly."""
        # Create curriculum
        arena = make_arena(num_agents=4)
        curriculum_config = env_curriculum(arena)
        curriculum = Curriculum(curriculum_config)

        # Generate some tasks and complete them
        for _ in range(5):
            task = curriculum.get_task()
            task.complete(0.5)  # Complete with some score

        # Save state
        original_state = curriculum.get_checkpoint_state()

        # Create new curriculum and load state
        new_curriculum = Curriculum(curriculum_config)
        new_curriculum.load_checkpoint_state(original_state)

        # Verify state matches
        assert new_curriculum._num_created == curriculum._num_created
        assert new_curriculum._num_evicted == curriculum._num_evicted
        assert len(new_curriculum._tasks) == len(curriculum._tasks)

        # Verify task details
        for task_id, task in curriculum._tasks.items():
            new_task = new_curriculum._tasks[task_id]
            assert new_task._task_id == task._task_id
            assert new_task._num_completions == task._num_completions
            assert new_task._total_score == task._total_score
            assert new_task._mean_score == task._mean_score
            assert new_task._num_scheduled == task._num_scheduled

    def test_checkpoint_manager_integration(self):
        """Test curriculum checkpointing through CheckpointManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(run="test_curriculum", run_dir=temp_dir)

            # Create and modify curriculum
            arena = make_arena(num_agents=4)
            curriculum = Curriculum(env_curriculum(arena))

            for _ in range(3):
                task = curriculum.get_task()
                task.complete(0.7)

            # Save through checkpoint manager
            checkpoint_manager.save_curriculum_state(curriculum)

            # Verify file exists
            curriculum_file = Path(temp_dir) / "test_curriculum" / "checkpoints" / "curriculum_state.pt"
            assert curriculum_file.exists()

            # Load through checkpoint manager
            loaded_state = checkpoint_manager.load_curriculum_state()
            assert loaded_state is not None

            # Create new curriculum and load state
            new_curriculum = Curriculum(env_curriculum(arena))
            new_curriculum.load_checkpoint_state(loaded_state)

            assert new_curriculum._num_created == curriculum._num_created

    def test_config_hash_validation(self):
        """Test that config hash detects configuration changes."""
        arena = make_arena(num_agents=4)
        curriculum_config = env_curriculum(arena)
        curriculum = Curriculum(curriculum_config)

        # Get original hash
        original_hash = curriculum._get_config_hash()

        # Create curriculum with different config
        different_arena = make_arena(num_agents=8)  # Different number of agents
        different_config = env_curriculum(different_arena)
        different_curriculum = Curriculum(different_config)

        # Hash should be different
        different_hash = different_curriculum._get_config_hash()
        assert original_hash != different_hash

    def test_empty_curriculum_checkpoint(self):
        """Test checkpointing an empty curriculum."""
        arena = make_arena(num_agents=4)
        curriculum_config = env_curriculum(arena)
        curriculum = Curriculum(curriculum_config)

        # Don't generate any tasks - just test basic state
        state = curriculum.get_checkpoint_state()

        # Create new curriculum and load state
        new_curriculum = Curriculum(curriculum_config)
        new_curriculum.load_checkpoint_state(state)

        # Should work without errors
        assert new_curriculum._num_created == curriculum._num_created
        assert new_curriculum._num_evicted == curriculum._num_evicted

    def test_task_tracker_checkpoint(self):
        """Test task tracker checkpointing separately."""
        from metta.cogworks.curriculum.task_tracker import TaskTracker

        tracker = TaskTracker(max_memory_tasks=100)

        # Add some task data
        tracker.track_task_creation(1)
        tracker.update_task_performance(1, 0.5)
        tracker.track_task_creation(2)
        tracker.update_task_performance(2, 0.8)

        # Save and load state
        state = tracker.get_checkpoint_state()
        new_tracker = TaskTracker(max_memory_tasks=100)
        new_tracker.load_checkpoint_state(state)

        # Verify data preserved
        task1_stats = new_tracker.get_task_stats(1)
        task2_stats = new_tracker.get_task_stats(2)

        assert task1_stats is not None
        assert task2_stats is not None
        assert task1_stats["completion_count"] == 1
        assert task2_stats["completion_count"] == 1

    def test_learning_progress_algorithm_checkpoint(self):
        """Test learning progress algorithm checkpointing with curriculum."""
        from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

        # Create curriculum with learning progress algorithm
        arena = make_arena(num_agents=4)
        curriculum_config = env_curriculum(arena)
        curriculum_config.num_active_tasks = 5  # Use smaller number for testing
        curriculum_config.algorithm_config = LearningProgressConfig(use_bidirectional=True)
        curriculum = Curriculum(curriculum_config)

        # Generate some tasks and complete them to build algorithm state
        for i in range(10):
            task = curriculum.get_task()
            task.complete(0.3 + (i % 3) * 0.2)  # Vary scores to build learning progress data
            curriculum.update_task_performance(task._task_id, task._mean_score)

        # Save state
        original_state = curriculum.get_checkpoint_state()

        # Create new curriculum and load state
        new_curriculum_config = env_curriculum(arena)
        new_curriculum_config.num_active_tasks = 5  # Match original
        new_curriculum_config.algorithm_config = LearningProgressConfig(use_bidirectional=True)
        new_curriculum = Curriculum(new_curriculum_config)
        new_curriculum.load_checkpoint_state(original_state)

        # Verify algorithm state is preserved
        assert new_curriculum._algorithm is not None
        assert type(new_curriculum._algorithm).__name__ == type(curriculum._algorithm).__name__

        # Verify task tracker data is preserved
        original_tracker_stats = curriculum._algorithm.task_tracker.get_global_stats()
        new_tracker_stats = new_curriculum._algorithm.task_tracker.get_global_stats()
        assert original_tracker_stats["total_tracked_tasks"] == new_tracker_stats["total_tracked_tasks"]

        # Also verify that the algorithm's bidirectional state is preserved
        if hasattr(curriculum._algorithm, "_outcomes") and curriculum._algorithm._outcomes:
            original_outcomes = len(curriculum._algorithm._outcomes)
            new_outcomes = len(new_curriculum._algorithm._outcomes)
            assert original_outcomes == new_outcomes
