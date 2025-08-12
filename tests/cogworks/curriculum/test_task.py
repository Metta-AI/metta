"""Tests for Task class."""

from cogworks.curriculum import Task
from metta.rl.env_config import EnvConfig


class TestTask:
    """Test cases for Task class."""

    def test_task_creation(self):
        """Test creating a Task with required parameters."""
        env_cfg = EnvConfig()
        task_id = "test_task"
        task = Task(task_id=task_id, env_cfg=env_cfg)

        assert task.env_cfg == env_cfg
        assert task.get_env_config() == env_cfg
        assert task.get_id() == task_id

    def test_task_with_custom_id(self):
        """Test creating a Task with different IDs."""
        env_cfg = EnvConfig()
        custom_id = "my_custom_task_42"
        task = Task(task_id=custom_id, env_cfg=env_cfg)

        assert task.env_cfg == env_cfg
        assert task.get_env_config() == env_cfg
        assert task.get_id() == custom_id

    def test_task_with_different_env_configs(self):
        """Test tasks with different env configs."""
        # Create env configs with different values
        env_cfg1 = EnvConfig(seed=1, device="cpu")
        env_cfg2 = EnvConfig(seed=2, device="cuda")

        task1 = Task(task_id="task1", env_cfg=env_cfg1)
        task2 = Task(task_id="task2", env_cfg=env_cfg2)

        assert task1.get_env_config() != task2.get_env_config()
        assert task1.get_id() != task2.get_id()

    def test_task_immutability_assumption(self):
        """Test that Task preserves env_config reference."""
        env_cfg = EnvConfig()
        task = Task(task_id="test", env_cfg=env_cfg)

        # Task should maintain reference to the same env_config object
        assert task.get_env_config() is env_cfg
        assert task.env_cfg is env_cfg

    def test_task_str_representation(self):
        """Test that task has reasonable string representation."""
        env_cfg = EnvConfig()
        task = Task(task_id="test_task", env_cfg=env_cfg)

        # Should be able to convert to string without error
        str_repr = str(task)
        assert isinstance(str_repr, str)

    def test_task_instances_are_unique(self):
        """Test that each Task instance is unique."""
        env_cfg = EnvConfig()

        task1 = Task(task_id="task1", env_cfg=env_cfg)
        task2 = Task(task_id="task1", env_cfg=env_cfg)  # Same ID and config
        task3 = Task(task_id="task2", env_cfg=env_cfg)

        # Each instance should be unique (no __eq__ override)
        assert task1 is not task2
        assert task1 is not task3

        # Can be used in sets - each instance is unique
        task_set = {task1, task2, task3}
        assert len(task_set) == 3  # All tasks are unique instances

        # Each has unique hash (default object hash)
        assert hash(task1) != hash(task2)
        assert hash(task1) != hash(task3)
