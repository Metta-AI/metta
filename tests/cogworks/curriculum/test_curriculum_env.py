"""Tests for CurriculumEnv."""

from unittest.mock import Mock

import numpy as np

from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    SingleTaskGeneratorConfig,
)
from metta.cogworks.curriculum.curriculum_env import CurriculumEnv
from metta.mettagrid.mettagrid_config import MettaGridConfig


class TestCurriculumEnv:
    """Test cases for CurriculumEnv."""

    def create_test_curriculum(self):
        """Helper to create a test curriculum."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=5, new_task_rate=0.1)
        return Curriculum(config, seed=0)

    def create_mock_env(self):
        """Helper to create a mock environment."""
        mock_env = Mock()
        mock_env.step.return_value = (
            np.array([1, 2, 3]),  # obs
            np.array([0.5, 0.7]),  # rewards
            np.array([False, False]),  # terminals
            np.array([False, False]),  # truncations
            {},  # infos
        )

        mock_env.get_episode_rewards = Mock(return_value=np.array([1.0, 2.0]))  # Add get_episode_rewards method
        mock_env.set_mg_config = Mock()  # Add set_mg_config method
        return mock_env

    def test_curriculum_env_creation(self):
        """Test creating a CurriculumEnv."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()

        wrapper = CurriculumEnv(mock_env, curriculum)

        assert wrapper._env is mock_env
        assert wrapper._curriculum is curriculum
        assert isinstance(wrapper._current_task, CurriculumTask)

    def test_curriculum_env_step_no_termination(self):
        """Test step method when episode doesn't terminate."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task

        # Step the environment
        result = wrapper.step([1, 0])

        # Should call env.step with correct args
        mock_env.step.assert_called_once_with([1, 0])

        # Should return the result from env.step
        expected = (np.array([1, 2, 3]), np.array([0.5, 0.7]), np.array([False, False]), np.array([False, False]), {})
        assert len(result) == len(expected)
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])
        np.testing.assert_array_equal(result[2], expected[2])
        np.testing.assert_array_equal(result[3], expected[3])
        assert result[4] == expected[4]

        # Task should remain the same (no termination)
        assert wrapper._current_task is initial_task

        # Should not have called set_env_cfg
        assert not mock_env.set_env_cfg.called

    def test_curriculum_env_step_with_termination(self):
        """Test step method when episode terminates."""
        mock_env = self.create_mock_env()
        # Set up termination condition
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.8, 0.9]),
            np.array([True, True]),  # Both terminated
            np.array([False, False]),
            {},
        )
        # Set up get_episode_rewards to return matching values
        mock_env.get_episode_rewards.return_value = np.array([0.8, 0.9])

        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task
        initial_completions = initial_task._num_completions

        # Step the environment
        _ = wrapper.step([1, 0])

        # Should call env.step
        mock_env.step.assert_called_once_with([1, 0])

        # Task should have been completed with mean reward
        assert initial_task._num_completions == initial_completions + 1
        assert abs(initial_task._total_score - 0.85) < 1e-10  # (0.8 + 0.9) / 2

        # Should have gotten a new task
        assert wrapper._current_task is not initial_task
        assert isinstance(wrapper._current_task, CurriculumTask)

        # Should have set new env config
        mock_env.set_mg_config.assert_called_once_with(wrapper._current_task.get_env_cfg())

    def test_curriculum_env_step_with_truncation(self):
        """Test step method when episode truncates."""
        mock_env = self.create_mock_env()
        # Set up truncation condition
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.6, 0.4]),
            np.array([False, False]),
            np.array([True, True]),  # Both truncated
            {},
        )
        # Set up get_episode_rewards to return matching values
        mock_env.get_episode_rewards.return_value = np.array([0.6, 0.4])

        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task

        # Step the environment
        wrapper.step([1, 0])

        # Task should have been completed
        assert initial_task._num_completions == 1
        assert initial_task._total_score == 0.5  # (0.6 + 0.4) / 2

        # Should have gotten a new task
        assert wrapper._current_task is not initial_task

    def test_curriculum_env_step_partial_termination(self):
        """Test step method when only some agents terminate."""
        mock_env = self.create_mock_env()
        # Set up partial termination
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.8, 0.2]),
            np.array([True, False]),  # Only first agent terminated
            np.array([False, False]),
            {},
        )

        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task

        # Step the environment
        wrapper.step([1, 0])

        # Task should remain the same (not all terminated)
        assert wrapper._current_task is initial_task
        assert initial_task._num_completions == 0

        # Should not have called set_env_cfg
        assert not mock_env.set_env_cfg.called

    def test_curriculum_env_step_partial_truncation(self):
        """Test step method when only some agents truncate."""
        mock_env = self.create_mock_env()
        # Set up partial truncation
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.3, 0.7]),
            np.array([False, False]),
            np.array([False, True]),  # Only second agent truncated
            {},
        )

        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task

        # Step the environment
        wrapper.step([1, 0])

        # Task should remain the same (not all truncated)
        assert wrapper._current_task is initial_task
        assert initial_task._num_completions == 0

    def test_curriculum_env_getattr_delegation(self):
        """Test that attribute access is delegated to the wrapped environment."""
        mock_env = self.create_mock_env()
        mock_env.some_attribute = "test_value"
        mock_env.some_method = Mock(return_value="method_result")

        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        # Should delegate attribute access
        assert wrapper.some_attribute == "test_value"

        # Should delegate method calls
        result = wrapper.some_method("arg1", kwarg="value")
        assert result == "method_result"
        mock_env.some_method.assert_called_once_with("arg1", kwarg="value")

    def test_curriculum_env_step_with_kwargs(self):
        """Test step method with keyword arguments."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        # Step with kwargs
        wrapper.step([1, 0], render=True, mode="human")

        # Should pass kwargs through
        mock_env.step.assert_called_once_with([1, 0], render=True, mode="human")

    def test_curriculum_env_multiple_episodes(self):
        """Test wrapper behavior across multiple episodes."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        tasks_seen = []

        for episode in range(3):
            # Record current task
            tasks_seen.append(wrapper._current_task)

            # Simulate episode ending
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                np.array([0.5 + episode * 0.1, 0.6 + episode * 0.1]),
                np.array([True, True]),
                np.array([False, False]),
                {},
            )
            # Set up get_episode_rewards to return matching values
            mock_env.get_episode_rewards.return_value = np.array([0.5 + episode * 0.1, 0.6 + episode * 0.1])

            wrapper.step([1, 0])

        # Should have seen 3 different tasks (one per episode)
        assert len(set(tasks_seen)) == 3

        # Each task should have been completed
        for task in tasks_seen:
            assert task._num_completions == 1

    def test_curriculum_env_reward_aggregation(self):
        """Test that rewards are properly aggregated for task completion."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        # Test with different reward arrays
        test_cases = [
            (np.array([1.0, 0.0]), 0.5),  # Simple average
            (np.array([0.8, 0.6, 0.4]), 0.6),  # Three agents
            (np.array([1.0]), 1.0),  # Single agent
            (np.array([-0.5, 0.5]), 0.0),  # Negative rewards
        ]

        for rewards, expected_mean in test_cases:
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                rewards,
                np.array([True] * len(rewards)),
                np.array([False] * len(rewards)),
                {},
            )
            # Set up get_episode_rewards to return matching values
            mock_env.get_episode_rewards.return_value = rewards

            initial_task = wrapper._current_task
            wrapper.step([1, 0])

            # Check that task was completed with correct mean reward
            assert initial_task._num_completions == 1
            assert abs(initial_task._total_score - expected_mean) < 1e-6


class TestCurriculumEnvEdgeCases:
    """Test edge cases and error conditions."""

    def test_curriculum_env_wrapper_zero_rewards(self):
        """Test wrapper behavior with zero rewards."""
        mock_env = Mock()
        # Simulate a 2-agent environment with zero rewards and no termination
        mock_env.step.return_value = (
            np.array([[1, 2, 3], [4, 5, 6]]),  # obs for 2 agents
            np.array([0.0, 0.0]),  # Zero rewards
            np.array([False, False]),  # No termination
            np.array([False, False]),  # No truncation
            {},
        )

        mock_env.get_episode_rewards = Mock(return_value=np.array([0.0, 0.0]))
        mock_env.set_mg_config = Mock()

        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config)
        curriculum = Curriculum(config, seed=0)

        wrapper = CurriculumEnv(mock_env, curriculum)
        initial_task = wrapper._current_task

        # Step with 2 agent actions
        result = wrapper.step([[0, 0], [0, 0]])
        assert len(result) == 5

        # Task should remain the same since no termination occurred
        assert wrapper._current_task is initial_task

    def test_curriculum_env_wrapper_single_agent(self):
        """Test wrapper with single-agent environment."""
        mock_env = Mock()
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.8]),  # Single reward
            np.array([True]),  # Single terminal
            np.array([False]),  # Single truncation
            {},
        )

        mock_env.get_episode_rewards = Mock(return_value=np.array([0.8]))
        mock_env.set_mg_config = Mock()

        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config)
        curriculum = Curriculum(config, seed=0)

        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task
        wrapper.step([1])

        # Should complete task with single reward
        assert initial_task._num_completions == 1
        assert initial_task._total_score == 0.8

    def test_curriculum_env_wrapper_curriculum_stats_integration(self):
        """Test that wrapper integrates properly with curriculum statistics."""
        mock_env = Mock()
        mock_env.step.return_value = (
            np.array([1, 2, 3]),
            np.array([0.7, 0.3]),
            np.array([True, True]),
            np.array([False, False]),
            {},
        )

        mock_env.get_episode_rewards = Mock(return_value=np.array([0.7, 0.3]))
        mock_env.set_mg_config = Mock()

        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=2)
        curriculum = Curriculum(config, seed=0)
        wrapper = CurriculumEnv(mock_env, curriculum)

        # Initial curriculum stats
        initial_stats = curriculum.stats()
        assert initial_stats["num_completed"] == 0
        assert initial_stats["num_scheduled"] == 1  # One task already created in wrapper

        # Complete an episode
        wrapper.step([1, 0])

        # Check updated stats
        updated_stats = curriculum.stats()
        assert updated_stats["num_completed"] == 1
        assert updated_stats["num_scheduled"] == 2  # Original + new task
