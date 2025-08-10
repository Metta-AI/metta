"""Tests for Task class."""

import pytest
from cogworks.curriculum.curriculum import Task
from metta.rl.env_config import EnvConfig


class TestTask:
    """Test cases for Task class."""

    def test_task_creation_with_defaults(self):
        """Test creating a Task with default values."""
        env_cfg = EnvConfig()
        task = Task(env_cfg)
        
        assert task.env_cfg == env_cfg
        assert task.get_env_config() == env_cfg
        assert isinstance(task.get_id(), str)
        assert len(task.get_id()) > 0
        
    def test_task_creation_with_custom_id(self):
        """Test creating a Task with a custom task_id."""
        env_cfg = EnvConfig()
        custom_id = "my_custom_task_42"
        task = Task(env_cfg, task_id=custom_id)
        
        assert task.env_cfg == env_cfg
        assert task.get_env_config() == env_cfg
        assert task.get_id() == custom_id
        
    def test_task_id_generation_from_env_config(self):
        """Test that task ID is generated deterministically from env_config."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        # Same env configs should generate same task IDs
        task1 = Task(env_cfg1)
        task2 = Task(env_cfg1)
        assert task1.get_id() == task2.get_id()
        
        # Different env configs might generate different task IDs
        task3 = Task(env_cfg2)
        # Note: IDs might be same due to default configs being identical
        
    def test_task_equality(self):
        """Test task equality based on env_config."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        task1a = Task(env_cfg1, task_id="task1")
        task1b = Task(env_cfg1, task_id="different_id")  # Same env_cfg, different ID
        task2 = Task(env_cfg2, task_id="task2")
        
        # Tasks are equal if they have the same env_config, regardless of task_id
        assert task1a == task1b
        assert task1a == Task(env_cfg1)
        
        # Tasks with different env_configs are not equal
        assert task1a != task2
        assert task1a != "not_a_task"
        assert task1a != None
        
    def test_task_hash(self):
        """Test task hashing for use in sets/dicts."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        task1a = Task(env_cfg1, task_id="task1")
        task1b = Task(env_cfg1, task_id="different_id")
        task2 = Task(env_cfg2, task_id="task2")
        
        # Tasks with same env_config should have same hash
        assert hash(task1a) == hash(task1b)
        
        # Can be used in sets
        task_set = {task1a, task1b, task2}
        # Should only contain unique env_configs
        assert len(task_set) <= 2  # Depends on whether env_cfg1 == env_cfg2
        
        # Can be used as dict keys
        task_dict = {task1a: "value1", task2: "value2"}
        assert task_dict[task1b] == "value1"  # Same env_config as task1a
        
    def test_task_with_different_env_configs(self):
        """Test tasks with meaningfully different env configs."""
        # Create env configs with different values
        env_cfg1 = EnvConfig(seed=1, device="cpu")
        env_cfg2 = EnvConfig(seed=2, device="cuda")
        
        task1 = Task(env_cfg1)
        task2 = Task(env_cfg2)
        
        assert task1 != task2
        assert hash(task1) != hash(task2)
        assert task1.get_env_config() != task2.get_env_config()
        
    def test_task_immutability_assumption(self):
        """Test that Task preserves env_config reference."""
        env_cfg = EnvConfig()
        task = Task(env_cfg)
        
        # Task should maintain reference to the same env_config object
        assert task.get_env_config() is env_cfg
        assert task.env_cfg is env_cfg
        
    def test_task_str_representation(self):
        """Test that task has reasonable string representation."""
        env_cfg = EnvConfig()
        task = Task(env_cfg, task_id="test_task")
        
        # Should be able to convert to string without error
        str_repr = str(task)
        assert isinstance(str_repr, str)
        
    def test_task_with_none_task_id(self):
        """Test task creation with None task_id (should generate one)."""
        env_cfg = EnvConfig()
        task = Task(env_cfg, task_id=None)
        
        # Should generate a task ID based on env_config
        assert task.get_id() is not None
        assert isinstance(task.get_id(), str)
        assert len(task.get_id()) > 0