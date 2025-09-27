"""
Tests for curriculum checkpointing functionality.

This module tests that curriculum state can be properly saved and loaded,
ensuring that training can be resumed correctly after restart.
"""

import pickle
import random
import tempfile

import pytest

from softmax.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from softmax.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from softmax.cogworks.curriculum.task_generator import SingleTaskGenerator
from softmax.cogworks.curriculum.task_tracker import TaskTracker
from softmax.training.rl.checkpoint_manager import CheckpointManager
from softmax.training.rl.system_config import SystemConfig
from mettagrid.config import GameConfig, MettaGridConfig


class TestCurriculumStateSerialization:
    """Test curriculum state serialization and deserialization."""

    def test_curriculum_state_serialization(self):
        """Test that curriculum state can be saved and loaded correctly."""
        # Create a curriculum config with learning progress algorithm
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
            system_config = SystemConfig(data_dir=temp_dir, local_only=True)
            checkpoint_manager = CheckpointManager(run="test_curriculum", system_cfg=system_config)

            # Create curriculum state
            mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
            task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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
            system_config = SystemConfig(data_dir=temp_dir, local_only=True)
            checkpoint_manager = CheckpointManager(run="test_backward_compat", system_cfg=system_config)

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
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

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

        # Verify that tasks can be retrieved and have proper data
        for task_id in original_tasks:
            if task_id in curriculum2._tasks:
                task = curriculum2._tasks[task_id]

                # Verify task data is preserved
                assert task._num_completions == original_tasks[task_id]["num_completions"]
                assert task._total_score == original_tasks[task_id]["total_score"]
                assert abs(task._mean_score - original_tasks[task_id]["mean_score"]) < 1e-6
                assert task._num_scheduled == original_tasks[task_id]["num_scheduled"]

        # Verify env_cfg is recreated when tasks are accessed
        retrieved_task = curriculum2.get_task()
        assert retrieved_task._env_cfg is not None, "env_cfg should be recreated when task is accessed"

    def test_lazy_env_cfg_recreation_performance(self):
        """Test that lazy env_cfg recreation improves loading performance."""
        import time

        # Create curriculum with many tasks
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=1000)

        curriculum = Curriculum(curriculum_config, seed=42)

        # Generate many tasks
        for _ in range(100):
            task = curriculum.get_task()
            task.complete(0.5)

        # Measure loading time
        state = curriculum.get_state()

        start_time = time.time()
        curriculum2 = Curriculum(curriculum_config, seed=999)
        curriculum2.load_state(state)
        load_time = time.time() - start_time

        # Loading should be fast (< 10 seconds for 1000 tasks)
        assert load_time < 10.0, f"Loading took {load_time:.2f}s, expected < 10.0s"

        # Verify tasks work correctly after loading
        task = curriculum2.get_task()
        assert task._env_cfg is not None, "env_cfg should be recreated when task is accessed"


class TestCurriculumRoundtripBehavior:
    """Test that curriculum behavior is preserved through save/load cycles."""

    def test_discrete_random_curriculum_roundtrip(self):
        """Test roundtrip for discrete random curriculum (no algorithm)."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=None,  # Use default discrete random
        )

        # Create original curriculum
        curriculum1 = Curriculum(curriculum_config, seed=42)

        # Simulate training activity
        task_selections = []
        for _ in range(30):
            task = curriculum1.get_task()
            task_selections.append(task._task_id)

            # Complete some tasks with scores
            if random.random() < 0.7:
                score = random.uniform(0.3, 0.9)
                task.complete(score)
                curriculum1.update_task_performance(task._task_id, score)

        # Checkpoint and restore
        state = curriculum1.get_state()
        curriculum2 = Curriculum(curriculum_config, seed=999)  # Different seed
        curriculum2.load_state(state)

        # Continue activity and verify consistent behavior
        post_restore_selections = []
        for _ in range(30):
            task = curriculum2.get_task()
            post_restore_selections.append(task._task_id)

            if random.random() < 0.7:
                score = random.uniform(0.3, 0.9)
                task.complete(score)
                curriculum2.update_task_performance(task._task_id, score)

        # Verify curriculum state consistency
        assert curriculum1._num_created == curriculum2._num_created
        assert curriculum1._num_evicted == curriculum2._num_evicted
        assert len(curriculum1._tasks) == len(curriculum2._tasks)

        # Tasks should exist and be selectable
        assert len(post_restore_selections) == 30
        assert all(task_id in curriculum2._tasks for task_id in post_restore_selections)

    def test_learning_progress_curriculum_roundtrip(self):
        """Test roundtrip for learning progress curriculum with bidirectional scoring."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=15,
            algorithm_config=LearningProgressConfig(
                num_active_tasks=15, use_bidirectional=True, max_memory_tasks=100, ema_timescale=0.01
            ),
        )

        # Create and train original curriculum
        curriculum1 = Curriculum(curriculum_config, seed=123)

        # Create diverse task performance to trigger learning progress
        task_performance = {}
        for _ in range(50):
            task = curriculum1.get_task()
            task_id = task._task_id

            # Create varied performance patterns
            if task_id not in task_performance:
                task_performance[task_id] = []

            # Simulate learning: improving scores over time for some tasks
            if len(task_performance[task_id]) < 3:
                score = random.uniform(0.2, 0.4)  # Initial low performance
            elif len(task_performance[task_id]) < 8:
                score = random.uniform(0.4, 0.7)  # Improving
            else:
                score = random.uniform(0.6, 0.9)  # Learned

            task.complete(score)
            curriculum1.update_task_performance(task_id, score)
            task_performance[task_id].append(score)

        # Capture algorithm state before checkpointing
        original_algorithm = curriculum1._algorithm
        original_task_scores = original_algorithm.score_tasks(list(curriculum1._tasks.keys()))
        original_stats = original_algorithm.stats()

        # Checkpoint and restore
        state = curriculum1.get_state()
        curriculum2 = Curriculum(curriculum_config, seed=456)  # Different seed
        curriculum2.load_state(state)

        # Verify algorithm state preservation
        restored_algorithm = curriculum2._algorithm
        restored_task_scores = restored_algorithm.score_tasks(list(curriculum2._tasks.keys()))
        restored_stats = restored_algorithm.stats()

        # Learning progress scores should be preserved
        assert len(original_task_scores) == len(restored_task_scores)
        for task_id in original_task_scores:
            if task_id in restored_task_scores:
                # Scores should be approximately equal (floating point tolerance)
                assert abs(original_task_scores[task_id] - restored_task_scores[task_id]) < 1e-6

        # Key statistics should be preserved
        key_stats = ["tracker/total_completions", "tracker/total_tracked_tasks"]
        for stat_key in key_stats:
            if stat_key in original_stats and stat_key in restored_stats:
                assert original_stats[stat_key] == restored_stats[stat_key]

        # Continue training and verify algorithm functionality
        for _ in range(20):
            task = curriculum2.get_task()
            score = random.uniform(0.5, 0.8)
            task.complete(score)
            curriculum2.update_task_performance(task._task_id, score)

        # Algorithm should still function (able to score tasks)
        final_scores = restored_algorithm.score_tasks(list(curriculum2._tasks.keys()))
        assert len(final_scores) > 0
        assert all(isinstance(score, (int, float)) for score in final_scores.values())

    def test_curriculum_deterministic_after_restore(self):
        """Test that curriculum produces deterministic sequences after restore."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=10,
            algorithm_config=LearningProgressConfig(num_active_tasks=10),
        )

        # Create two identical curricula
        curriculum1 = Curriculum(curriculum_config, seed=789)
        curriculum2 = Curriculum(curriculum_config, seed=789)

        # Run identical training on both
        for _ in range(25):
            task1 = curriculum1.get_task()
            task2 = curriculum2.get_task()

            # Should select same tasks (deterministic)
            assert task1._task_id == task2._task_id

            score = random.uniform(0.4, 0.8)
            task1.complete(score)
            task2.complete(score)
            curriculum1.update_task_performance(task1._task_id, score)
            curriculum2.update_task_performance(task2._task_id, score)

        # Checkpoint curriculum1 and restore to curriculum3
        state = curriculum1.get_state()
        curriculum3 = Curriculum(curriculum_config, seed=999)  # Different seed
        curriculum3.load_state(state)

        # Continue with curriculum2 and curriculum3 - they should be identical
        for _ in range(15):
            task2 = curriculum2.get_task()
            task3 = curriculum3.get_task()

            # Should produce same sequence after restore
            assert task2._task_id == task3._task_id

            score = random.uniform(0.5, 0.9)
            task2.complete(score)
            task3.complete(score)
            curriculum2.update_task_performance(task2._task_id, score)
            curriculum3.update_task_performance(task3._task_id, score)

    def test_task_state_consistency_after_roundtrip(self):
        """Test that task internal state is consistent after save/load."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(
            task_generator=task_generator_config,
            num_active_tasks=8,
            algorithm_config=LearningProgressConfig(num_active_tasks=8, use_bidirectional=True),
        )

        curriculum = Curriculum(curriculum_config, seed=333)

        # Build up task history
        task_history = {}
        for _ in range(40):
            task = curriculum.get_task()
            task_id = task._task_id

            if task_id not in task_history:
                task_history[task_id] = {"scores": [], "scheduled_count": 0}

            task_history[task_id]["scheduled_count"] += 1

            # Complete task with score
            if random.random() < 0.8:
                score = random.uniform(0.2, 0.95)
                task.complete(score)
                curriculum.update_task_performance(task_id, score)
                task_history[task_id]["scores"].append(score)

        # Verify pre-checkpoint state
        pre_checkpoint_tasks = {}
        for task_id, task in curriculum._tasks.items():
            pre_checkpoint_tasks[task_id] = {
                "num_completions": task._num_completions,
                "total_score": task._total_score,
                "mean_score": task._mean_score,
                "num_scheduled": task._num_scheduled,
            }

        # Checkpoint and restore
        state = curriculum.get_state()
        curriculum2 = Curriculum(curriculum_config, seed=777)
        curriculum2.load_state(state)

        # Verify all task states are identical
        for task_id, expected_state in pre_checkpoint_tasks.items():
            assert task_id in curriculum2._tasks
            restored_task = curriculum2._tasks[task_id]

            assert restored_task._num_completions == expected_state["num_completions"]
            assert abs(restored_task._total_score - expected_state["total_score"]) < 1e-10
            assert abs(restored_task._mean_score - expected_state["mean_score"]) < 1e-10
            assert restored_task._num_scheduled == expected_state["num_scheduled"]

        # Verify task functionality post-restore
        for _ in range(10):
            # Get task (this increments num_scheduled for the returned task)
            task = curriculum2.get_task()
            initial_completions = task._num_completions
            initial_total_score = task._total_score

            # Task should still be functional
            score = 0.75
            task.complete(score)
            curriculum2.update_task_performance(task._task_id, score)

            # State should update correctly after completion
            assert task._num_completions == initial_completions + 1
            assert task._total_score == initial_total_score + score
            assert task._num_scheduled >= 1  # Should be at least 1 since we just got it

    def test_empty_curriculum_roundtrip(self):
        """Test roundtrip with minimal/empty curriculum state."""
        mg_config = MettaGridConfig(game=GameConfig(num_agents=4))
        task_generator_config = SingleTaskGenerator.Config(env=mg_config)

        curriculum_config = CurriculumConfig(task_generator=task_generator_config, num_active_tasks=5)

        # Create curriculum but don't use it much
        curriculum1 = Curriculum(curriculum_config, seed=444)

        # Just create the initial task pool, no additional activity
        assert len(curriculum1._tasks) == 5  # Should be initialized at capacity

        # Checkpoint immediately
        state = curriculum1.get_state()
        curriculum2 = Curriculum(curriculum_config, seed=555)
        curriculum2.load_state(state)

        # Should restore successfully
        assert len(curriculum2._tasks) == len(curriculum1._tasks)
        assert curriculum2._num_created == curriculum1._num_created
        assert curriculum2._num_evicted == curriculum1._num_evicted

        # Should be functional after restore
        task = curriculum2.get_task()
        assert task is not None
        assert task._task_id in curriculum2._tasks
