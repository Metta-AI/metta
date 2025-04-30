import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import DictConfig

from mettagrid.curriculum.curriculum import (
    LowRewardCurriculum,
    MettaGridCurriculum,
    MettaGridTask,
    MultiEnvCurriculum,
    Task,
)


class TestTask(unittest.TestCase):
    def setUp(self):
        self.curriculum = MagicMock()
        self.task = Task("test_task", self.curriculum)

    def test_initial_state(self):
        self.assertFalse(self.task.is_complete())
        self.assertEqual(self.task._id, "test_task")

    def test_complete(self):
        self.task.complete(0.5)
        self.assertTrue(self.task.is_complete())
        self.curriculum.complete_task.assert_called_once_with("test_task", 0.5)

    def test_complete_twice_raises(self):
        self.task.complete(0.5)
        with self.assertRaises(AssertionError):
            self.task.complete(0.6)


class TestMettaGridTask(unittest.TestCase):
    def setUp(self):
        self.curriculum = MagicMock()
        self.game_cfg = DictConfig(
            {"num_agents": 2, "map_builder": {"type": "test_builder"}, "recursive_map_builder": True}
        )
        self.game_map = np.array([["agent1", "empty"], ["empty", "agent2"]])

    def test_initialization_with_map(self):
        task = MettaGridTask("test_task", self.curriculum, self.game_cfg, self.game_map)
        self.assertEqual(task._id, "test_task")
        self.assertEqual(task._game_cfg, self.game_cfg)
        np.testing.assert_array_equal(task._level_map, self.game_map)

    @patch("mettagrid.curriculum.curriculum.simple_instantiate")
    def test_initialization_without_map(self, mock_instantiate):
        mock_builder = MagicMock()
        mock_builder.build.return_value = self.game_map
        mock_instantiate.return_value = mock_builder

        task = MettaGridTask("test_task", self.curriculum, self.game_cfg)
        mock_instantiate.assert_called_once()
        np.testing.assert_array_equal(task._level_map, self.game_map)

    def test_complete(self):
        task = MettaGridTask("test_task", self.curriculum, self.game_cfg, self.game_map)
        infos = {"episode_rewards": np.array([1.0, 2.0])}
        task.complete(infos)
        self.assertTrue(task.is_complete())
        self.curriculum.complete_task.assert_called_once_with("test_task", 3.0)


class TestMettaGridCurriculum(unittest.TestCase):
    def setUp(self):
        self.game_cfg = DictConfig({"num_agents": 2, "max_steps": 100})

    def test_get_task(self):
        curriculum = MettaGridCurriculum(self.game_cfg)
        task = curriculum.get_task()
        self.assertIsInstance(task, MettaGridTask)
        self.assertEqual(task._id, "default")
        self.assertEqual(task._game_cfg, self.game_cfg)

    def test_complete_task(self):
        curriculum = MettaGridCurriculum(self.game_cfg)
        curriculum.complete_task("test_task", 0.5)
        # Just verify no errors are raised


class TestMultiEnvCurriculum(unittest.TestCase):
    def setUp(self):
        self.tasks = {"task1": 0.6, "task2": 0.4}
        self.game_cfg = DictConfig({"num_agents": 2, "max_steps": 100})

    @patch("mettagrid.curriculum.curriculum.config_from_path")
    def test_get_task(self, mock_config_from_path):
        mock_config = MagicMock()
        mock_config.game = self.game_cfg
        mock_config_from_path.return_value = mock_config

        curriculum = MultiEnvCurriculum(self.tasks, self.game_cfg)
        task = curriculum.get_task()
        self.assertIsInstance(task, MettaGridTask)
        self.assertIn(task._id, ["task1", "task2"])

    def test_complete_task(self):
        curriculum = MultiEnvCurriculum(self.tasks, self.game_cfg)
        curriculum.complete_task("test_task", 0.5)
        # Just verify no errors are raised


class TestLowRewardCurriculum(unittest.TestCase):
    def setUp(self):
        self.tasks = {"task1": 0.6, "task2": 0.4}
        self.game_cfg = DictConfig({"num_agents": 2, "max_steps": 100})

    @patch("mettagrid.curriculum.curriculum.config_from_path")
    def test_get_task_adapts_weights(self, mock_config_from_path):
        mock_config = MagicMock()
        mock_config.game = self.game_cfg
        mock_config_from_path.return_value = mock_config

        curriculum = LowRewardCurriculum(self.tasks, self.game_cfg)

        # Complete task1 with low reward
        curriculum.complete_task("task1", 0.1)

        # Complete task2 with high reward
        curriculum.complete_task("task2", 0.9)

        # Next task should be biased towards task1
        task = curriculum.get_task()
        self.assertEqual(task._id, "task1")

    def test_reward_averages_update(self):
        curriculum = LowRewardCurriculum(self.tasks, self.game_cfg)

        # Initial averages should be 0
        self.assertEqual(curriculum._reward_averages["task1"], 0.0)
        self.assertEqual(curriculum._reward_averages["task2"], 0.0)

        # Update task1 reward
        curriculum.complete_task("task1", 0.5)
        self.assertGreater(curriculum._reward_averages["task1"], 0.0)
        self.assertEqual(curriculum._reward_averages["task2"], 0.0)


if __name__ == "__main__":
    unittest.main()
