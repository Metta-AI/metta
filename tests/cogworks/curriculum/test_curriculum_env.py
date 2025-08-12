"""Tests for the CurriculumEnv wrapper."""

from unittest.mock import Mock

import numpy as np

from metta.cogworks.curriculum.curriculum import CurriculumTask
from metta.cogworks.curriculum.curriculum_env import CurriculumEnv

from .conftest import create_test_env_config


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self):
        self.step_count = 0
        self.reset_count = 0
        self.env_cfg = None

    def step(self, *args, **kwargs):
        """Mock step method."""
        self.step_count += 1
        # Return typical step output: obs, rewards, terminals, truncations, infos
        obs = np.array([1, 2, 3])
        rewards = np.array([0.5, 0.3])
        terminals = np.array([False, False])
        truncations = np.array([False, False])
        infos = {}
        return obs, rewards, terminals, truncations, infos

    def step_with_termination(self, *args, **kwargs):
        """Mock step method that returns termination."""
        self.step_count += 1
        obs = np.array([1, 2, 3])
        rewards = np.array([0.8, 0.6])
        terminals = np.array([True, True])
        truncations = np.array([False, False])
        infos = {}
        return obs, rewards, terminals, truncations, infos

    def step_with_truncation(self, *args, **kwargs):
        """Mock step method that returns truncation."""
        self.step_count += 1
        obs = np.array([1, 2, 3])
        rewards = np.array([0.2, 0.4])
        terminals = np.array([False, False])
        truncations = np.array([True, True])
        infos = {}
        return obs, rewards, terminals, truncations, infos

    def step_with_empty_rewards(self, *args, **kwargs):
        """Mock step method that returns empty rewards."""
        self.step_count += 1
        obs = np.array([1, 2, 3])
        rewards = np.array([])  # Empty rewards
        terminals = np.array([True])
        truncations = np.array([False])
        infos = {}
        return obs, rewards, terminals, truncations, infos

    def reset(self):
        """Mock reset method."""
        self.reset_count += 1
        return np.array([0, 0, 0])

    def set_env_cfg(self, env_cfg):
        """Mock method for setting environment config."""
        self.env_cfg = env_cfg

    def some_other_method(self):
        """Mock method for testing attribute delegation."""
        return "delegated"


class MockCurriculum:
    """Mock curriculum for testing."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []

    def get_task(self):
        """Mock get_task method."""
        self.task_count += 1
        env_cfg = create_test_env_config(seed=self.task_count)
        task = CurriculumTask(task_id=self.task_count, env_cfg=env_cfg)

        # Override complete method to track completions
        original_complete = task.complete

        def tracked_complete(score):
            self.completed_tasks.append((self.task_count, score))
            return original_complete(score)

        task.complete = tracked_complete

        return task


class TestCurriculumEnv:
    """Test cases for CurriculumEnv."""

    def test_init(self):
        """Test CurriculumEnv initialization."""
        env = MockEnvironment()
        curriculum = MockCurriculum()

        curriculum_env = CurriculumEnv(env, curriculum)

        assert curriculum_env._env == env
        assert curriculum_env._curriculum == curriculum
        assert curriculum_env._current_task is not None
        assert curriculum.task_count == 1  # One task created during init

    def test_step_no_termination(self):
        """Test step method without episode termination."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        initial_task = curriculum_env._current_task
        initial_task_count = curriculum.task_count

        # Step without termination
        obs, rewards, terminals, truncations, infos = curriculum_env.step()

        assert np.array_equal(obs, np.array([1, 2, 3]))
        assert np.array_equal(rewards, np.array([0.5, 0.3]))
        assert np.array_equal(terminals, np.array([False, False]))
        assert np.array_equal(truncations, np.array([False, False]))
        assert env.step_count == 1

        # Task should not change, no completion
        assert curriculum_env._current_task == initial_task
        assert curriculum.task_count == initial_task_count
        assert len(curriculum.completed_tasks) == 0

    def test_step_with_termination(self):
        """Test step method with episode termination."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        initial_task = curriculum_env._current_task
        initial_task_count = curriculum.task_count

        # Override step to return termination
        env.step = env.step_with_termination

        # Step with termination
        obs, rewards, terminals, truncations, infos = curriculum_env.step()

        assert np.array_equal(obs, np.array([1, 2, 3]))
        assert np.array_equal(rewards, np.array([0.8, 0.6]))
        assert np.array_equal(terminals, np.array([True, True]))
        assert env.step_count == 1

        # Task should change and be completed
        assert curriculum_env._current_task != initial_task
        assert curriculum.task_count == initial_task_count + 1  # New task created
        assert len(curriculum.completed_tasks) == 1
        assert curriculum.completed_tasks[0][1] == 0.7  # Mean of [0.8, 0.6]

        # Environment should have new config set
        assert env.env_cfg is not None

    def test_step_with_truncation(self):
        """Test step method with episode truncation."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        initial_task = curriculum_env._current_task

        # Override step to return truncation
        env.step = env.step_with_truncation

        # Step with truncation
        curriculum_env.step()

        # Task should change and be completed (truncation counts as completion)
        assert curriculum_env._current_task != initial_task
        assert len(curriculum.completed_tasks) == 1
        assert abs(curriculum.completed_tasks[0][1] - 0.3) < 1e-10  # Mean of [0.2, 0.4]

    def test_step_with_empty_rewards(self):
        """Test step method with empty rewards array."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        initial_task = curriculum_env._current_task

        # Override step to return empty rewards
        env.step = env.step_with_empty_rewards

        # Step with empty rewards and termination
        curriculum_env.step()

        # Task should change and be completed with 0.0 score
        assert curriculum_env._current_task != initial_task
        assert len(curriculum.completed_tasks) == 1
        assert curriculum.completed_tasks[0][1] == 0.0  # Empty rewards should give 0.0

    def test_step_partial_termination(self):
        """Test step method with partial termination/truncation."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        initial_task = curriculum_env._current_task

        def partial_termination_step(*args, **kwargs):
            """Mock step with partial termination."""
            env.step_count += 1
            obs = np.array([1, 2, 3])
            rewards = np.array([0.5, 0.7, 0.9])
            terminals = np.array([True, False, True])  # Partial termination
            truncations = np.array([False, False, False])
            infos = {}
            return obs, rewards, terminals, truncations, infos

        env.step = partial_termination_step

        # Step with partial termination - should NOT complete task
        curriculum_env.step()

        # Task should NOT change since not all agents terminated
        assert curriculum_env._current_task == initial_task
        assert len(curriculum.completed_tasks) == 0

    def test_attribute_delegation(self):
        """Test that unknown attributes are delegated to wrapped environment."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        # Test delegation of method
        result = curriculum_env.some_other_method()
        assert result == "delegated"

        # Test delegation of attribute
        assert curriculum_env.step_count == env.step_count
        assert curriculum_env.reset_count == env.reset_count

    def test_step_with_args_and_kwargs(self):
        """Test that step passes through arguments correctly."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        # Mock step to capture arguments
        original_step = env.step
        env.step = Mock(return_value=original_step())

        # Call step with various arguments
        curriculum_env.step(1, 2, action="move", direction="north")

        # Verify arguments were passed through
        env.step.assert_called_once_with(1, 2, action="move", direction="north")

    def test_task_scheduling_tracking(self):
        """Test that tasks are properly scheduled when obtained."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        # Initial task should have been scheduled once
        initial_task = curriculum_env._current_task
        assert initial_task._num_scheduled == 0  # Not scheduled yet in our mock

        # Force task completion to get a new task
        env.step = env.step_with_termination
        curriculum_env.step()

        # New task should be available
        new_task = curriculum_env._current_task
        assert new_task != initial_task

    def test_multiple_episode_completions(self):
        """Test multiple episode completions in sequence."""
        env = MockEnvironment()
        curriculum = MockCurriculum()
        curriculum_env = CurriculumEnv(env, curriculum)

        env.step = env.step_with_termination

        # Complete multiple episodes
        scores = []
        for _i in range(3):
            curriculum_env.step()
            if curriculum.completed_tasks:
                scores.append(curriculum.completed_tasks[-1][1])

        assert len(curriculum.completed_tasks) == 3
        assert len(scores) == 3
        # All scores should be the same since we're using the same mock step
        assert all(score == 0.7 for score in scores)

        # Should have created 4 tasks total (initial + 3 new ones)
        assert curriculum.task_count == 4
