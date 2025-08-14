"""Test task-scaled performance functionality."""

from unittest.mock import Mock

import numpy as np

from metta.cogworks.curriculum.curriculum import CurriculumTask
from metta.cogworks.curriculum.curriculum_env import CurriculumEnv
from metta.cogworks.curriculum.task_generator import (
    BucketedTaskGeneratorConfig,
    SingleTaskGeneratorConfig,
    ValueRange,
)
from metta.mettagrid.mettagrid_config import EnvConfig


def test_reward_target_in_env_config():
    """Test that reward_target can be set in EnvConfig."""
    config = EnvConfig()
    assert config.reward_target is None

    config.reward_target = 10.0
    assert config.reward_target == 10.0


def test_bucketed_generator_with_reward_target():
    """Test that BucketedTaskGenerator can set reward_target."""
    # Create a simple task generator config
    child_config = SingleTaskGeneratorConfig(
        type="single",
        env_config=EnvConfig(),
    )

    # Create bucketed config with reward target bucket and a dummy bucket
    config = BucketedTaskGeneratorConfig(
        type="bucketed",
        child_generator_config=child_config,
        buckets={"game.num_agents": [1]},  # Dummy bucket to satisfy non-empty requirement
        reward_target_bucket=[5.0, 10.0, 15.0],
    )

    generator = config.create()

    # Generate a task
    task = generator.get_task(42)

    # Check that reward_target is set
    assert task.reward_target is not None
    assert task.reward_target in [5.0, 10.0, 15.0]


def test_reward_target_calculation():
    """Test task-scaled performance calculation."""
    # Test cases: (reward, target, expected_scaled_performance)
    test_cases = [
        (5.0, 10.0, 0.5),  # 50% performance
        (10.0, 10.0, 1.0),  # 100% performance
        (15.0, 10.0, 1.0),  # Capped at 100%
        (0.0, 10.0, 0.0),  # 0% performance
        (2.5, 5.0, 0.5),  # 50% performance
    ]

    for reward, target, expected in test_cases:
        scaled_performance = min(reward / target, 1.0)
        assert scaled_performance == expected, (
            f"Expected {expected}, got {scaled_performance} for reward={reward}, target={target}"
        )


def test_value_range_reward_target():
    """Test using ValueRange for continuous reward target sampling."""
    child_config = SingleTaskGeneratorConfig(
        type="single",
        env_config=EnvConfig(),
    )

    config = BucketedTaskGeneratorConfig(
        type="bucketed",
        child_generator_config=child_config,
        buckets={"game.max_steps": [100]},  # Dummy bucket to satisfy non-empty requirement
        reward_target_bucket=[ValueRange.vr(5.0, 15.0)],
    )

    generator = config.create()

    # Generate multiple tasks to test sampling
    targets = []
    for i in range(10):
        task = generator.get_task(i)
        targets.append(task.reward_target)

    # All targets should be within the range
    for target in targets:
        assert 5.0 <= target <= 15.0


def test_backward_compatibility():
    """Test that existing configurations work without reward_target."""
    child_config = SingleTaskGeneratorConfig(
        type="single",
        env_config=EnvConfig(),
    )

    config = BucketedTaskGeneratorConfig(
        type="bucketed",
        child_generator_config=child_config,
        buckets={"game.num_agents": [1]},  # Dummy bucket to satisfy non-empty requirement
        # No reward_target_bucket specified
    )

    generator = config.create()
    task = generator.get_task(42)

    # Should work without reward_target
    assert task.reward_target is None


def test_curriculum_env_task_scaled_performance_calculation():
    """Test that CurriculumEnv correctly calculates task-scaled performance."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards (50% of target)
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has reward_target
    env_config = EnvConfig()
    env_config.reward_target = 10.0

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was calculated and logged
    assert "task_scaled_performance" in infos
    assert 42 in infos["task_scaled_performance"]
    assert infos["task_scaled_performance"][42] == 0.5  # 5.0 / 10.0 = 0.5


def test_curriculum_env_no_reward_target():
    """Test that CurriculumEnv handles tasks without reward_target gracefully."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has NO reward_target
    env_config = EnvConfig()
    env_config.reward_target = None  # No reward target

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was NOT calculated
    assert "task_scaled_performance" not in infos


def test_curriculum_env_zero_reward_target():
    """Test that CurriculumEnv handles zero reward_target gracefully."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has zero reward_target
    env_config = EnvConfig()
    env_config.reward_target = 0.0  # Zero reward target

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was NOT calculated (division by zero avoided)
    assert "task_scaled_performance" not in infos


def test_curriculum_env_negative_reward_target():
    """Test that CurriculumEnv handles negative reward_target gracefully."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has negative reward_target
    env_config = EnvConfig()
    env_config.reward_target = -10.0  # Negative reward target

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was NOT calculated (negative target avoided)
    assert "task_scaled_performance" not in infos


def test_curriculum_env_performance_capping():
    """Test that task-scaled performance is correctly capped at 1.0."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([15.0]),  # rewards (150% of target)
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has reward_target
    env_config = EnvConfig()
    env_config.reward_target = 10.0

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was capped at 1.0
    assert "task_scaled_performance" in infos
    assert 42 in infos["task_scaled_performance"]
    assert infos["task_scaled_performance"][42] == 1.0  # Capped at 1.0


def test_curriculum_env_multiple_episodes():
    """Test that task-scaled performance is calculated for multiple episodes."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards
        np.array([True]),  # terminals
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create tasks with different reward targets
    env_config_1 = EnvConfig()
    env_config_1.reward_target = 10.0
    task_1 = CurriculumTask(task_id=42, env_cfg=env_config_1)

    env_config_2 = EnvConfig()
    env_config_2.reward_target = 20.0
    task_2 = CurriculumTask(task_id=43, env_cfg=env_config_2)

    # Create a mock curriculum that returns different tasks
    mock_curriculum = Mock()
    mock_curriculum.get_task.side_effect = [task_1, task_2, task_1]  # Add more values to side_effect

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task_1

    # First episode
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])
    assert "task_scaled_performance" in infos
    assert 42 in infos["task_scaled_performance"]
    assert infos["task_scaled_performance"][42] == 0.5  # 5.0 / 10.0 = 0.5

    # Second episode (new task)
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])
    assert "task_scaled_performance" in infos
    assert 43 in infos["task_scaled_performance"]
    assert infos["task_scaled_performance"][43] == 0.25  # 5.0 / 20.0 = 0.25


def test_curriculum_env_empty_rewards():
    """Test that CurriculumEnv handles empty rewards gracefully."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([]),  # empty obs
        np.array([]),  # empty rewards
        np.array([True]),  # terminals (episode done, but empty rewards)
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has reward_target
    env_config = EnvConfig()
    env_config.reward_target = 10.0

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([])

    # Check that task-scaled performance was calculated with mean reward of 0.0
    # Note: Empty rewards result in mean_reward = 0.0, so scaled performance should be 0.0
    assert "task_scaled_performance" in infos
    assert 42 in infos["task_scaled_performance"]
    assert infos["task_scaled_performance"][42] == 0.0  # 0.0 / 10.0 = 0.0


def test_task_generator_deterministic_sampling():
    """Test that reward target sampling is deterministic for the same task_id."""
    child_config = SingleTaskGeneratorConfig(
        type="single",
        env_config=EnvConfig(),
    )

    config = BucketedTaskGeneratorConfig(
        type="bucketed",
        child_generator_config=child_config,
        buckets={"game.num_agents": [1]},
        reward_target_bucket=[5.0, 10.0, 15.0, 20.0, 25.0],
    )

    generator = config.create()

    # Generate the same task multiple times
    task_1 = generator.get_task(42)
    task_2 = generator.get_task(42)
    task_3 = generator.get_task(42)

    # All should have the same reward_target (deterministic)
    assert task_1.reward_target == task_2.reward_target
    assert task_2.reward_target == task_3.reward_target

    # Different task_id should have different reward_target
    task_4 = generator.get_task(43)
    assert task_1.reward_target != task_4.reward_target


def test_task_generator_fixed_reward_target():
    """Test that fixed reward_target overrides reward_target_bucket."""
    child_config = SingleTaskGeneratorConfig(
        type="single",
        env_config=EnvConfig(),
    )

    config = BucketedTaskGeneratorConfig(
        type="bucketed",
        child_generator_config=child_config,
        buckets={"game.num_agents": [1]},
        reward_target=7.5,  # Fixed reward target
        # No reward_target_bucket specified - should use fixed value
    )

    generator = config.create()

    # Generate multiple tasks
    for i in range(5):
        task = generator.get_task(i)
        assert task.reward_target == 7.5  # Should always be the fixed value


def test_curriculum_env_non_episode_completion():
    """Test that task-scaled performance is not calculated for non-completed episodes."""
    # Create a mock environment
    mock_env = Mock()
    mock_env.step.return_value = (
        np.array([[1, 2, 3]]),  # obs
        np.array([5.0]),  # rewards
        np.array([False]),  # terminals (episode not done)
        np.array([False]),  # truncations
        {},  # infos
    )
    mock_env.set_env_cfg = Mock()

    # Create a curriculum with a task that has reward_target
    env_config = EnvConfig()
    env_config.reward_target = 10.0

    task = CurriculumTask(task_id=42, env_cfg=env_config)

    # Create a mock curriculum
    mock_curriculum = Mock()
    mock_curriculum.get_task.return_value = task

    # Create CurriculumEnv
    curriculum_env = CurriculumEnv(mock_env, mock_curriculum)
    curriculum_env._current_task = task

    # Step the environment
    obs, rewards, terminals, truncations, infos = curriculum_env.step([0])

    # Check that task-scaled performance was NOT calculated (episode not done)
    assert "task_scaled_performance" not in infos
