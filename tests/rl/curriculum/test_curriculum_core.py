#!/usr/bin/env python3
"""Core curriculum functionality tests."""

from omegaconf import OmegaConf

from tests.rl.curriculum.conftest import MockCurriculum, StatefulCurriculum


class TestCurriculumCore:
    """Test core curriculum functionality."""

    def test_curriculum_basic_operations(self):
        """Test basic curriculum operations: get_task, complete_task."""
        curriculum = MockCurriculum()
        
        # Test get_task
        task1 = curriculum.get_task()
        assert task1.name() == "task_1"
        assert task1.env_cfg() is not None
        assert task1.env_cfg().game.num_agents == 1
        
        task2 = curriculum.get_task()
        assert task2.name() == "task_2"
        
        # Test complete_task
        curriculum.complete_task("task_1", 0.8)
        assert len(curriculum.completed_tasks) == 1
        assert curriculum.completed_tasks[0] == ("task_1", 0.8)

    def test_curriculum_stats_methods(self):
        """Test that curriculum stats methods work correctly."""
        curriculum = StatefulCurriculum()

        # Generate some tasks
        for _ in range(5):
            curriculum.get_task()

        # Simulate completions
        curriculum.complete_task("easy_task_1", 0.8)
        curriculum.complete_task("easy_task_2", 0.9)
        curriculum.complete_task("hard_task_3", 0.6)

        # Test stats
        stats = curriculum.stats()
        assert stats["total_tasks"] == 5
        assert stats["completed_tasks"] == 3
        assert abs(stats["average_score"] - 0.767) < 0.01  # (0.8 + 0.9 + 0.6) / 3

        # Test task probabilities
        task_probs = curriculum.get_task_probs()
        assert abs(task_probs["easy"] - 0.65) < 0.01  # Adjusted from 0.7 due to high scores
        assert abs(task_probs["hard"] - 0.35) < 0.01  # Adjusted from 0.3

        # Test completion rates
        completion_rates = curriculum.get_completion_rates()
        assert completion_rates["task_completions/easy"] == 2/3
        assert completion_rates["task_completions/hard"] == 1/3

    def test_curriculum_task_config_variation(self):
        """Test that curriculum generates tasks with varying configurations."""
        curriculum = StatefulCurriculum()
        
        configs = []
        for _ in range(6):
            task = curriculum.get_task()
            configs.append(task.env_cfg())
        
        # Should have both easy and hard tasks
        easy_configs = [c for c in configs if c.game.difficulty == "easy"]
        hard_configs = [c for c in configs if c.game.difficulty == "hard"]
        
        assert len(easy_configs) > 0
        assert len(hard_configs) > 0
        
        # Easy and hard should have different sizes
        assert easy_configs[0].game.width == 10
        assert hard_configs[0].game.width == 20

    def test_curriculum_learning_adaptation(self):
        """Test that curriculum adapts based on performance."""
        curriculum = StatefulCurriculum()
        
        initial_probs = curriculum.get_task_probs().copy()
        
        # Complete several easy tasks with high scores
        for _ in range(3):
            task = curriculum.get_task()
            if "easy" in task.name():
                curriculum.complete_task(task.name(), 0.9)
        
        # Task probabilities should have shifted
        new_probs = curriculum.get_task_probs()
        assert new_probs["easy"] < initial_probs["easy"]
        assert new_probs["hard"] > initial_probs["hard"]

    def test_task_interface(self):
        """Test Task class interface."""
        curriculum = MockCurriculum()
        task = curriculum.get_task()
        
        # Test task methods
        assert hasattr(task, "name")
        assert hasattr(task, "id")
        assert hasattr(task, "env_cfg")
        assert hasattr(task, "complete")
        
        # Test calling methods
        assert task.name() == "task_1"
        assert task.id() == "task_1"
        cfg = task.env_cfg()
        assert isinstance(cfg, (dict, OmegaConf)) or hasattr(cfg, '__getitem__')
        
        # Test complete method
        task.complete(0.85)
        assert curriculum.completed_tasks[-1] == ("task_1", 0.85)