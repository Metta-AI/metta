"""Tests for Curriculum classes."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cogworks.curriculum.curriculum import Curriculum, RandomCurriculum, LearningProgressCurriculum, Task
from cogworks.curriculum.config import (
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
    WeightedTaskSetConfig,
    WeightedTaskSetItem
)
from metta.rl.env_config import EnvConfig


class TestCurriculumBase:
    """Test cases for Curriculum base class."""
    
    def test_curriculum_is_abstract(self):
        """Test that Curriculum cannot be instantiated directly."""
        config = CurriculumConfig(task_set_config=WeightedTaskSetConfig(items=[]))
        
        with pytest.raises(TypeError):
            Curriculum(config)
            
    def test_curriculum_base_methods(self):
        """Test default implementations of base methods."""
        # Create a concrete subclass for testing
        class TestCurriculum(Curriculum):
            def get_task(self, seed: int) -> Task:
                env_cfg = EnvConfig()
                return Task(env_cfg, task_id=f"test_{seed}")
                
        config = CurriculumConfig(task_set_config=WeightedTaskSetConfig(items=[]))
        curriculum = TestCurriculum(config, seed=0)
        
        # Test default implementations
        task = Task(EnvConfig())
        curriculum.complete_task(task, 0.5)  # Should not raise
        
        assert curriculum.get_task_probs() == {}
        assert curriculum.get_curriculum_stats() == {}


class TestRandomCurriculum:
    """Test cases for RandomCurriculum."""
    
    def create_test_config(self):
        """Helper to create a test configuration."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item])
        return RandomCurriculumConfig(task_set_config=task_set_config)
        
    def test_random_curriculum_creation(self):
        """Test creating a RandomCurriculum."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)
        
        assert curriculum.config == config
        assert hasattr(curriculum, 'task_set')
        
    def test_random_curriculum_get_task_deterministic(self):
        """Test that same seed produces same task."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)
        
        task1 = curriculum.get_task(42)
        task2 = curriculum.get_task(42)  # Same seed
        task3 = curriculum.get_task(43)  # Different seed
        
        # Same seed should produce same task
        assert task1.get_id() == task2.get_id()
        assert task1 == task2
        
        # Different seed should produce different task ID
        assert task1.get_id() != task3.get_id()
        
        # All tasks should have consistent format
        assert task1.get_id().startswith("random_42")
        assert task3.get_id().startswith("random_43")
        
    def test_random_curriculum_task_content(self):
        """Test that tasks contain expected env config."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)
        
        task = curriculum.get_task(42)
        env_cfg = task.get_env_config()
        
        # Should match the configured env config
        
    def test_random_curriculum_multiple_items(self):
        """Test RandomCurriculum with multiple weighted items."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=1.0)
        item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item1, item2])
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        
        curriculum = RandomCurriculum(config, seed=0)
        
        # Sample multiple tasks
        tasks = [curriculum.get_task(i) for i in range(10)]
        
        # All tasks should be valid
        for task in tasks:
            assert task is not None
            assert hasattr(task, 'get_env_config')
            
    def test_random_curriculum_complete_task(self):
        """Test that complete_task doesn't crash."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)
        
        task = curriculum.get_task(42)
        
        # Should not raise any exception
        curriculum.complete_task(task, 0.8)
        curriculum.complete_task(task, 0.0)
        curriculum.complete_task(task, 1.0)


class TestLearningProgressCurriculum:
    """Test cases for LearningProgressCurriculum."""
    
    def create_test_config(self, n_tasks=5):
        """Helper to create a test configuration."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=1.0)
        item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item1, item2])
        
        return LearningProgressCurriculumConfig(
            task_set_config=task_set_config,
            n_tasks=n_tasks,
            num_active_tasks=3,
            memory=10,
            ema_timescale=0.1,
            progress_smoothing=0.1,
            rand_task_rate=0.2
        )
        
    def test_lp_curriculum_creation(self):
        """Test creating a LearningProgressCurriculum."""
        config = self.create_test_config()
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        assert curriculum.config == config
        assert len(curriculum.tasks) == 0  # Not initialized yet
        assert curriculum._tasks_initialized is False
        
    def test_lp_curriculum_task_initialization(self):
        """Test that tasks are initialized on first get_task call."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        # First call should initialize tasks
        task = curriculum.get_task(42)
        
        assert curriculum._tasks_initialized is True
        assert len(curriculum.tasks) == 3
        assert len(curriculum.task_outcomes) == 3
        assert len(curriculum.task_weights) == 3
        assert len(curriculum.active_task_indices) == 3  # min(3, num_active_tasks)
        
        # Task should be one of the initialized tasks
        assert task in curriculum.tasks
        
    def test_lp_curriculum_deterministic_task_generation(self):
        """Test that tasks are generated deterministically."""
        config = self.create_test_config(n_tasks=3)
        curriculum1 = LearningProgressCurriculum(config, seed=0)
        curriculum2 = LearningProgressCurriculum(config, seed=0)
        
        # Initialize both with same seed
        task1 = curriculum1.get_task(42)
        task2 = curriculum2.get_task(42)
        
        # Should generate the same set of tasks
        assert len(curriculum1.tasks) == len(curriculum2.tasks)
        for t1, t2 in zip(curriculum1.tasks, curriculum2.tasks):
            assert t1.get_id() == t2.get_id()
            
    def test_lp_curriculum_task_completion_updates(self):
        """Test that completing tasks updates learning progress."""
        config = self.create_test_config(n_tasks=2)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        # Initialize
        task = curriculum.get_task(42)
        
        # Complete some tasks
        curriculum.complete_task(curriculum.tasks[0], 0.8)
        curriculum.complete_task(curriculum.tasks[0], 0.9)
        curriculum.complete_task(curriculum.tasks[1], 0.2)
        
        # Check that outcomes were recorded
        task0_outcomes = curriculum.task_outcomes[curriculum.tasks[0].get_id()]
        task1_outcomes = curriculum.task_outcomes[curriculum.tasks[1].get_id()]
        
        assert len(task0_outcomes) == 2
        assert task0_outcomes == [0.8, 0.9]
        assert len(task1_outcomes) == 1
        assert task1_outcomes == [0.2]
        
    def test_lp_curriculum_memory_limit(self):
        """Test that task outcomes respect memory limit."""
        config = self.create_test_config(n_tasks=1)
        config.memory = 2  # Only keep 2 recent outcomes
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        task = curriculum.get_task(42)
        test_task = curriculum.tasks[0]
        
        # Add more outcomes than memory limit
        curriculum.complete_task(test_task, 0.1)
        curriculum.complete_task(test_task, 0.2)
        curriculum.complete_task(test_task, 0.3)
        curriculum.complete_task(test_task, 0.4)
        
        # Should only keep the most recent ones
        outcomes = curriculum.task_outcomes[test_task.get_id()]
        assert len(outcomes) == 2
        assert outcomes == [0.3, 0.4]
        
    def test_lp_curriculum_score_clamping(self):
        """Test that scores are clamped to [0, 1] range."""
        config = self.create_test_config(n_tasks=1)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        task = curriculum.get_task(42)
        test_task = curriculum.tasks[0]
        
        # Test extreme scores
        curriculum.complete_task(test_task, -10.0)  # Should be clamped to 0
        curriculum.complete_task(test_task, 10.0)   # Should be clamped to 1
        
        outcomes = curriculum.task_outcomes[test_task.get_id()]
        assert outcomes == [0.0, 1.0]
        
    def test_lp_curriculum_unknown_task_handling(self):
        """Test handling of unknown tasks in complete_task."""
        config = self.create_test_config(n_tasks=1)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        curriculum.get_task(42)  # Initialize
        
        # Create a task not in the curriculum
        unknown_task = Task(EnvConfig(), task_id="unknown")
        
        # Should log warning but not crash
        with patch('cogworks.curriculum.curriculum.logger') as mock_logger:
            curriculum.complete_task(unknown_task, 0.5)
            mock_logger.warning.assert_called_once()
            
    def test_lp_curriculum_learning_progress_updates(self):
        """Test that learning progress tracking is updated."""
        config = self.create_test_config(n_tasks=2)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        curriculum.get_task(42)  # Initialize
        
        # Initially, tracking variables should be None
        assert curriculum.p_fast is None
        assert curriculum.p_slow is None
        assert curriculum.p_true is None
        
        # Complete a task to trigger updates
        curriculum.complete_task(curriculum.tasks[0], 0.8)
        
        # Should initialize tracking variables
        assert curriculum.p_fast is not None
        assert curriculum.p_slow is not None
        assert curriculum.p_true is not None
        assert isinstance(curriculum.p_fast, np.ndarray)
        assert len(curriculum.p_fast) == 2
        
    def test_lp_curriculum_task_probs(self):
        """Test get_task_probs returns correct probabilities."""
        config = self.create_test_config(n_tasks=3)
        config.num_active_tasks = 2  # Only 2 active tasks
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        curriculum.get_task(42)  # Initialize
        
        probs = curriculum.get_task_probs()
        
        # Should have probabilities for all tasks
        assert len(probs) == 3
        
        # Only active tasks should have non-zero probabilities
        non_zero_probs = [p for p in probs.values() if p > 0]
        assert len(non_zero_probs) <= 2  # At most num_active_tasks
        
    def test_lp_curriculum_stats(self):
        """Test get_curriculum_stats returns meaningful stats."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        # Before initialization
        stats = curriculum.get_curriculum_stats()
        assert stats == {}
        
        # After initialization and some progress
        curriculum.get_task(42)
        curriculum.complete_task(curriculum.tasks[0], 0.8)
        
        stats = curriculum.get_curriculum_stats()
        
        # Should contain expected stats
        assert "lp/num_active_tasks" in stats
        assert "lp/mean_task_weight" in stats
        assert "lp/num_zero_weight_tasks" in stats
        assert "lp/mean_success_rate" in stats
        assert "lp/min_success_rate" in stats
        assert "lp/max_success_rate" in stats
        
        assert isinstance(stats["lp/num_active_tasks"], int)
        assert isinstance(stats["lp/mean_task_weight"], float)
        
    def test_lp_curriculum_weighted_sampling(self):
        """Test that tasks are sampled according to weights."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        curriculum.get_task(42)  # Initialize
        
        # Manually set task weights to test sampling
        curriculum.task_weights = np.array([0.8, 0.1, 0.1])
        curriculum.active_task_indices = [0, 1, 2]
        
        # Sample many tasks
        task_counts = {}
        for seed in range(100):
            task = curriculum.get_task(seed)
            task_id = task.get_id()
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
        # Task 0 should be selected more often due to higher weight
        task0_id = curriculum.tasks[0].get_id()
        assert task_counts[task0_id] > task_counts.get(curriculum.tasks[1].get_id(), 0)
        
    def test_lp_curriculum_no_active_tasks_fallback(self):
        """Test behavior when no active tasks are available."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        curriculum.get_task(42)  # Initialize
        
        # Remove all active tasks
        curriculum.active_task_indices = []
        
        # Should still be able to get a task (random fallback)
        task = curriculum.get_task(123)
        assert task in curriculum.tasks


class TestCurriculumEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_curriculum_with_empty_taskset(self):
        """Test curriculum behavior with empty task set."""
        task_set_config = WeightedTaskSetConfig(items=[])
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        curriculum = RandomCurriculum(config, seed=0)
        
        # Should raise error when trying to get task
        with pytest.raises(ValueError, match="No items to sample from"):
            curriculum.get_task(42)
            
    def test_lp_curriculum_zero_tasks(self):
        """Test LearningProgressCurriculum validation rejects n_tasks=0."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item])
        
        # Should reject n_tasks=0 during validation
        with pytest.raises(ValueError):
            LearningProgressCurriculumConfig(
                task_set_config=task_set_config,
                n_tasks=0,  # Zero tasks - should be invalid
                num_active_tasks=1
            )
        
    def test_lp_curriculum_more_active_than_total(self):
        """Test when num_active_tasks > n_tasks."""
        config = TestLearningProgressCurriculum().create_test_config(n_tasks=2)
        config.num_active_tasks = 5  # More than n_tasks
        
        curriculum = LearningProgressCurriculum(config, seed=0)
        curriculum.get_task(42)  # Initialize
        
        # Should limit to actual number of tasks
        assert len(curriculum.active_task_indices) == 2
        
    def test_curriculum_task_id_consistency(self):
        """Test that task IDs are consistent across curriculum instances."""
        config = TestLearningProgressCurriculum().create_test_config(n_tasks=2)
        
        curriculum1 = LearningProgressCurriculum(config, seed=0)
        curriculum2 = LearningProgressCurriculum(config, seed=0)
        
        # Same seed should produce same task set
        curriculum1.get_task(42)
        curriculum2.get_task(42)
        
        task_ids1 = [task.get_id() for task in curriculum1.tasks]
        task_ids2 = [task.get_id() for task in curriculum2.tasks]
        
        assert task_ids1 == task_ids2