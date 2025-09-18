"""
Tests for curriculum checkpointing functionality.

This module tests that curriculum state can be properly saved and loaded,
ensuring that training can be resumed correctly after restart.
"""

import pickle
import random
import tempfile

import pytest

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig
from metta.cogworks.curriculum.task_tracker import TaskTracker
from metta.mettagrid.mettagrid_config import GameConfig, MettaGridConfig
from metta.rl.checkpoint_manager import CheckpointManager


class TestCurriculumStateSerialization:
    """Test curriculum state serialization and deserialization."""

    def test_curriculum_state_serialization(self):
        """Test that curriculum state can be saved and loaded correctly."""
        # Create a curriculum config with learning progress algorithm
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=LearningProgressConfig(num_active_tasks=10, max_memory_tasks=100, use_bidirectional=True),
        )

        # Create curriculum with fixed seed for reproducibility
        curriculum = Curriculum(curriculum_config, seed=42)

        # Generate some tasks and performance data
        task_ids = []
        for _ in range(20):
            task = curriculum.get_task()
            task_ids.append(task._task_id)
            # Complete some tasks with random scores
            if random.random() < 0.7:  # 70% completion rate
                score = random.uniform(0.0, 1.0)
                task.complete(score)
                curriculum.update_task_performance(task._task_id, score)

        # Save state
        state = curriculum.get_state()

        # Verify state structure
        assert "config" in state
        assert "seed" in state
        assert "num_created" in state
        assert "num_evicted" in state
        assert "tasks" in state
        assert "algorithm_state" in state

        # Verify task data is saved
        assert len(state["tasks"]) > 0
        for _task_id, task_data in state["tasks"].items():
            assert "num_completions" in task_data
            assert "total_score" in task_data
            assert "mean_score" in task_data
            assert "num_scheduled" in task_data
            assert "slice_values" in task_data

        # Verify algorithm state is saved
        algorithm_state = state["algorithm_state"]
        assert algorithm_state["type"] == "learning_progress"
        assert "task_tracker" in algorithm_state
        assert "outcomes" in algorithm_state or "hypers" in algorithm_state

    def test_curriculum_state_loading(self):
        """Test that curriculum state can be loaded correctly."""
        # Create initial curriculum
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=LearningProgressConfig(num_active_tasks=10),
        )

        curriculum1 = Curriculum(curriculum_config, seed=42)

        # Generate tasks and performance data
        original_tasks = {}
        for _ in range(15):
            task = curriculum1.get_task()
            task_id = task._task_id
            original_tasks[task_id] = {
                "num_completions": task._num_completions,
                "total_score": task._total_score,
                "mean_score": task._mean_score,
                "num_scheduled": task._num_scheduled,
            }

            if random.random() < 0.6:
                score = random.uniform(0.0, 1.0)
                task.complete(score)
                curriculum1.update_task_performance(task_id, score)

        # Save state
        state = curriculum1.get_state()

        # Create new curriculum with different seed and load state
        curriculum2 = Curriculum(curriculum_config, seed=999)  # Different seed
        curriculum2.load_state(state)

        # Verify state restoration
        assert curriculum1._num_created == curriculum2._num_created
        assert curriculum1._num_evicted == curriculum2._num_evicted
        assert len(curriculum1._tasks) == len(curriculum2._tasks)
        assert curriculum1._task_ids == curriculum2._task_ids

        # Verify task data is restored correctly
        for task_id in original_tasks:
            if task_id in curriculum2._tasks:
                task1 = curriculum1._tasks[task_id]
                task2 = curriculum2._tasks[task_id]
                assert task1._num_completions == task2._num_completions
                assert task1._total_score == task2._total_score
                assert abs(task1._mean_score - task2._mean_score) < 1e-6
                assert task1._num_scheduled == task2._num_scheduled

    def test_learning_progress_algorithm_state(self):
        """Test LearningProgressAlgorithm state serialization."""
        # Create curriculum with learning progress algorithm
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=8,
            algorithm_config=LearningProgressConfig(num_active_tasks=8, use_bidirectional=True, max_memory_tasks=50),
        )

        curriculum = Curriculum(curriculum_config, seed=123)

        # Generate tasks and performance data to populate algorithm state
        for _ in range(20):
            task = curriculum.get_task()
            if random.random() < 0.8:
                score = random.uniform(0.0, 1.0)
                task.complete(score)
                curriculum.update_task_performance(task._task_id, score)

        # Get algorithm state
        algorithm = curriculum._algorithm
        state = algorithm.get_state()

        # Verify state structure
        assert state["type"] == "learning_progress"
        assert "task_tracker" in state
        assert "outcomes" in state
        assert "counter" in state
        assert "p_fast" in state
        assert "p_slow" in state

        # Verify task tracker state
        task_tracker_state = state["task_tracker"]
        assert "max_memory_tasks" in task_tracker_state
        assert "task_memory" in task_tracker_state
        assert "completion_history" in task_tracker_state

        # Test loading state
        new_algorithm = curriculum_config.algorithm_config.create(8)
        new_algorithm.load_state(state)

        # Verify state was loaded correctly
        assert len(algorithm.task_tracker._task_memory) == len(new_algorithm.task_tracker._task_memory)
        assert algorithm.task_tracker._cached_total_completions == new_algorithm.task_tracker._cached_total_completions

    def test_task_tracker_state(self):
        """Test TaskTracker state serialization."""
        tracker = TaskTracker(max_memory_tasks=100)

        # Add some tasks and performance data
        for i in range(10):
            tracker.track_task_creation(i)
            for _ in range(random.randint(1, 5)):
                score = random.uniform(0.0, 1.0)
                tracker.update_task_performance(i, score)

        # Get state
        state = tracker.get_state()

        # Verify state structure
        assert "max_memory_tasks" in state
        assert "task_memory" in state
        assert "task_creation_order" in state
        assert "completion_history" in state
        assert "cached_total_completions" in state
        assert "cache_valid" in state

        # Test loading state
        new_tracker = TaskTracker(max_memory_tasks=50)
        new_tracker.load_state(state)

        # Verify state was loaded correctly
        assert len(tracker._task_memory) == len(new_tracker._task_memory)
        assert tracker._cached_total_completions == new_tracker._cached_total_completions
        assert len(tracker._completion_history) == len(new_tracker._completion_history)

    def test_file_size_limits(self):
        """Test that checkpoint files don't become too large."""
        # Create curriculum with many tasks
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=1000,
            algorithm_config=LearningProgressConfig(num_active_tasks=1000, max_memory_tasks=5000),
        )

        curriculum = Curriculum(curriculum_config, seed=456)

        # Generate many tasks with performance data
        for _ in range(500):
            task = curriculum.get_task()
            if random.random() < 0.7:
                score = random.uniform(0.0, 1.0)
                task.complete(score)
                curriculum.update_task_performance(task._task_id, score)

        # Get state and measure size
        state = curriculum.get_state()
        serialized_size = len(pickle.dumps(state))

        # Assert file size is reasonable (less than 10MB)
        assert serialized_size < 10 * 1024 * 1024, f"State size {serialized_size} bytes exceeds 10MB limit"

        # Print size for reference
        print(f"Curriculum state size: {serialized_size / 1024 / 1024:.2f} MB")

    def test_config_mismatch_handling(self):
        """Test handling of config mismatches during restore."""
        # Create initial curriculum
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config1 = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=10)

        curriculum = Curriculum(curriculum_config1, seed=42)

        # Generate some tasks
        for _ in range(5):
            task = curriculum.get_task()
            task.complete(0.5)

        # Save state
        state = curriculum.get_state()

        # Create curriculum with different config
        curriculum_config2 = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=20,  # Different number of active tasks
        )

        curriculum2 = Curriculum(curriculum_config2, seed=789)

        # Loading should work even with config mismatch
        # The warning will be logged but we just verify loading succeeds
        curriculum2.load_state(state)

        # Verify that the state was loaded successfully
        assert curriculum2._num_created == curriculum._num_created
        assert len(curriculum2._tasks) == len(curriculum._tasks)

    def test_corrupted_state_handling(self):
        """Test handling of corrupted or incomplete state."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=10)

        curriculum = Curriculum(curriculum_config, seed=42)

        # Create corrupted state (missing required fields)
        corrupted_state = {
            "config": curriculum_config.model_dump(),
            "seed": curriculum._rng.getstate(),
            # Missing num_created, num_evicted, tasks
        }

        # Loading should raise an error
        with pytest.raises(KeyError):
            curriculum.load_state(corrupted_state)

    def test_random_state_preservation(self):
        """Test that random state is preserved correctly."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=10)

        curriculum = Curriculum(curriculum_config, seed=123)

        # Generate some tasks to advance random state
        for _ in range(10):
            curriculum.get_task()

        # Save state
        state = curriculum.get_state()

        # Create new curriculum and load state
        curriculum2 = Curriculum(curriculum_config, seed=0)  # Different initial seed
        curriculum2.load_state(state)

        # Generate more tasks and verify they're deterministic
        tasks1 = [curriculum.get_task()._task_id for _ in range(5)]
        tasks2 = [curriculum2.get_task()._task_id for _ in range(5)]

        assert tasks1 == tasks2, "Random state not preserved correctly"


class TestCheckpointManagerIntegration:
    """Test integration with CheckpointManager."""

    def test_checkpoint_manager_curriculum_state(self):
        """Test that CheckpointManager can save and load curriculum state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(run="test_curriculum", run_dir=temp_dir)

            # Create curriculum state
            mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
            task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

            curriculum_config = CurriculumConfig(
                task_generator=task_generator_config,
                num_active_tasks=10,
                algorithm_config=LearningProgressConfig(num_active_tasks=10),
            )

            curriculum = Curriculum(curriculum_config, seed=42)

            # Generate some tasks and performance data
            for _ in range(10):
                task = curriculum.get_task()
                if random.random() < 0.6:
                    score = random.uniform(0.0, 1.0)
                    task.complete(score)
                    curriculum.update_task_performance(task._task_id, score)

            # Get curriculum state
            curriculum_state = curriculum.get_state()

            # Save trainer state with curriculum
            import torch

            optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)])

            checkpoint_manager.save_trainer_state(
                optimizer=optimizer, epoch=5, agent_step=1000, curriculum_state=curriculum_state
            )

            # Load trainer state
            loaded_state = checkpoint_manager.load_trainer_state()

            assert loaded_state is not None
            assert "curriculum_state" in loaded_state
            assert loaded_state["epoch"] == 5
            assert loaded_state["agent_step"] == 1000

            # Verify curriculum state is preserved
            loaded_curriculum_state = loaded_state["curriculum_state"]
            assert loaded_curriculum_state["num_created"] == curriculum_state["num_created"]
            assert loaded_curriculum_state["num_evicted"] == curriculum_state["num_evicted"]
            assert len(loaded_curriculum_state["tasks"]) == len(curriculum_state["tasks"])

    def test_checkpoint_manager_without_curriculum_state(self):
        """Test CheckpointManager works without curriculum state (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(run="test_backward_compat", run_dir=temp_dir)

            # Save trainer state without curriculum
            import torch

            optimizer = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)])

            checkpoint_manager.save_trainer_state(optimizer=optimizer, epoch=3, agent_step=500)

            # Load trainer state
            loaded_state = checkpoint_manager.load_trainer_state()

            assert loaded_state is not None
            assert "curriculum_state" not in loaded_state
            assert loaded_state["epoch"] == 3
            assert loaded_state["agent_step"] == 500


class TestTaskRecreation:
    """Test that tasks are properly recreated after loading state."""

    def test_task_recreation_with_env_cfg(self):
        """Test that tasks are recreated with proper env_cfg after loading."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGeneratorConfig(env=mg_config)

        curriculum_config = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=10)

        curriculum = Curriculum(curriculum_config, seed=42)

        # Generate some tasks
        original_tasks = {}
        for _ in range(8):
            task = curriculum.get_task()
            task_id = task._task_id
            original_tasks[task_id] = {
                "num_completions": task._num_completions,
                "total_score": task._total_score,
                "mean_score": task._mean_score,
                "num_scheduled": task._num_scheduled,
                "slice_values": task._slice_values.copy(),
            }

        # Save and load state
        state = curriculum.get_state()
        curriculum2 = Curriculum(curriculum_config, seed=999)
        curriculum2.load_state(state)

        # Verify that tasks can be retrieved and have proper env_cfg
        for task_id in original_tasks:
            if task_id in curriculum2._tasks:
                task = curriculum2._tasks[task_id]
                # After loading, the task should have recreated env_cfg
                assert task._env_cfg is not None

                # Verify task data is preserved
                assert task._num_completions == original_tasks[task_id]["num_completions"]
                assert task._total_score == original_tasks[task_id]["total_score"]
                assert abs(task._mean_score - original_tasks[task_id]["mean_score"]) < 1e-6
                assert task._num_scheduled == original_tasks[task_id]["num_scheduled"]
