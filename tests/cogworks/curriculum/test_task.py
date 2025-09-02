"""Tests for Task class."""

from metta.cogworks.curriculum import Task
from metta.mettagrid.mettagrid_config import GameConfig, MettaGridConfig


class TestTask:
    """Test cases for Task class."""

    def test_task_creation(self):
        """Test creating a Task with required parameters."""
        cfg = MettaGridConfig()
        task_id = "test_task"
        task = Task(task_id=task_id, env_cfg=cfg)

        assert task.env_cfg == cfg
        assert task.get_mg_config() == cfg
        assert task.get_id() == task_id

    def test_task_with_custom_id(self):
        """Test creating a Task with different IDs."""
        cfg = MettaGridConfig()
        custom_id = "my_custom_task_42"
        task = Task(task_id=custom_id, env_cfg=cfg)

        assert task.env_cfg == cfg
        assert task.get_mg_config() == cfg
        assert task.get_id() == custom_id

    def test_task_with_different_mg_configs(self):
        """Test tasks with different env configs."""
        # Create env configs with different values
        cfg1 = MettaGridConfig(game=GameConfig(num_agents=1))
        cfg2 = MettaGridConfig(game=GameConfig(num_agents=2))

        task1 = Task(task_id="task1", env_cfg=cfg1)
        task2 = Task(task_id="task2", env_cfg=cfg2)

        assert task1.get_mg_config() != task2.get_mg_config()
        assert task1.get_id() != task2.get_id()

    def test_task_immutability_assumption(self):
        """Test that Task preserves mg_config reference."""
        cfg = MettaGridConfig()
        task = Task(task_id="test", env_cfg=cfg)

        # Task should maintain reference to the same mg_config object
        assert task.get_mg_config() is cfg
        assert task.env_cfg is cfg

    def test_task_str_representation(self):
        """Test that task has reasonable string representation."""
        cfg = MettaGridConfig()
        task = Task(task_id="test_task", env_cfg=cfg)

        # Should be able to convert to string without error
        str_repr = str(task)
        assert isinstance(str_repr, str)

    def test_task_instances_are_unique(self):
        """Test that each Task instance is unique."""
        cfg = MettaGridConfig()

        task1 = Task(task_id="task1", env_cfg=cfg)
        task2 = Task(task_id="task1", env_cfg=cfg)  # Same ID and config
        task3 = Task(task_id="task2", env_cfg=cfg)

        # Each instance should be unique (no __eq__ override)
        assert task1 is not task2
        assert task1 is not task3

        # Can be used in sets - each instance is unique
        task_set = {task1, task2, task3}
        assert len(task_set) == 3  # All tasks are unique instances

        # Each has unique hash (default object hash)
        assert hash(task1) != hash(task2)
        assert hash(task1) != hash(task3)
