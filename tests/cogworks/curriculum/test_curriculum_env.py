"""Simplified tests for CurriculumEnv using parameterization and helpers."""

from unittest.mock import Mock

import numpy as np
import pytest

from metta.cogworks.curriculum import Curriculum, CurriculumConfig, CurriculumTask, SingleTaskGenerator
from metta.cogworks.curriculum.curriculum_env import CurriculumEnv
from mettagrid.config.mettagrid_config import MettaGridConfig

from .test_helpers import CurriculumTestHelper


class TestCurriculumEnv:
    """Test cases for CurriculumEnv."""

    def create_test_curriculum(self):
        """Helper to create a test curriculum."""
        task_gen_config = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=5, seed=0)
        return Curriculum(config)

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

    @pytest.mark.parametrize(
        "termination_type,expected_behavior",
        [
            ("no_termination", "same_task"),
            ("full_termination", "new_task"),
            ("partial_termination", "new_task"),  # Now correctly handles partial completions!
            ("truncation", "new_task"),
        ],
    )
    def test_curriculum_env_termination_handling(self, termination_type, expected_behavior):
        """Test all termination scenarios in one parameterized test."""
        mock_env = self.create_mock_env()
        curriculum = self.create_test_curriculum()
        wrapper = CurriculumEnv(mock_env, curriculum)

        initial_task = wrapper._current_task

        # Set up termination condition based on type
        if termination_type == "no_termination":
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                np.array([0.5, 0.7]),
                np.array([False, False]),  # No termination
                np.array([False, False]),  # No truncation
                {},
            )
            mock_env.get_episode_rewards.return_value = np.array([0.5, 0.7])
        elif termination_type == "full_termination":
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                np.array([0.8, 0.9]),
                np.array([True, True]),  # Both terminated
                np.array([False, False]),
                {},
            )
            mock_env.get_episode_rewards.return_value = np.array([0.8, 0.9])
        elif termination_type == "partial_termination":
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                np.array([0.8, 0.2]),
                np.array([True, False]),  # Only first agent terminated
                np.array([False, False]),
                {},
            )
            mock_env.get_episode_rewards.return_value = np.array([0.8, 0.2])
        elif termination_type == "truncation":
            mock_env.step.return_value = (
                np.array([1, 2, 3]),
                np.array([0.6, 0.4]),
                np.array([False, False]),
                np.array([True, True]),  # Both truncated
                {},
            )
            mock_env.get_episode_rewards.return_value = np.array([0.6, 0.4])

        # Step the environment
        result = wrapper.step([1, 0])

        # Verify step result
        expected = mock_env.step.return_value
        CurriculumTestHelper.assert_step_result(result, expected)

        # Verify task completion behavior
        if expected_behavior == "same_task":
            assert wrapper._current_task is initial_task
            assert initial_task._num_completions == 0  # No environments completed
        else:  # new_task
            assert wrapper._current_task is not initial_task
            assert initial_task._num_completions > 0
            if termination_type == "full_termination":
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

        # Step with appropriate number of agent actions
        rewards = np.array([0.6, 0.4])  # From mock setup above
        if len(rewards) == 1:
            wrapper.step([1])
        else:
            wrapper.step([0] * len(rewards))

        # Check that task was completed with correct total and count (2 envs)
        expected_sum = np.sum(rewards)  # 1.0 (0.6 + 0.4)
        assert initial_task._num_completions == 2  # 2 environments completed
        assert abs(initial_task._total_score - expected_sum) < 1e-6
        # Mean should still be 0.5
        assert abs(initial_task._total_score / initial_task._num_completions - 0.5) < 1e-6

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
            mock_env.get_episode_rewards.return_value = np.array([0.5 + episode * 0.1, 0.6 + episode * 0.1])

            wrapper.step([1, 0])

        # Should have seen 3 different tasks (one per episode)
        assert len(set(tasks_seen)) == 3

        # Each task should have been completed once per vectorized environment (2 envs)
        for task in tasks_seen:
            assert task._num_completions == 2  # 2 environments completed each time

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

        task_gen_config = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, seed=0)
        curriculum = Curriculum(config)

        wrapper = CurriculumEnv(mock_env, curriculum)
        initial_task = wrapper._current_task

        # Step with 2 agent actions
        result = wrapper.step([0, 0])
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

        task_gen_config = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, seed=0)
        curriculum = Curriculum(config)

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

        task_gen_config = SingleTaskGenerator.Config(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=2, seed=0)
        curriculum = Curriculum(config)
        wrapper = CurriculumEnv(mock_env, curriculum)

        # Initial curriculum stats
        initial_stats = curriculum.stats()
        assert initial_stats["num_completed"] == 0
        assert initial_stats["num_scheduled"] == 1  # One task already created in wrapper

        # Complete an episode
        wrapper.step([1, 0])

        # Check updated stats (2 completions because 2 vectorized envs completed)
        updated_stats = curriculum.stats()
        assert updated_stats["num_completed"] == 2  # 2 environments completed
        assert updated_stats["num_scheduled"] == 2  # Original + new task
