"""Tests for the Task class."""

from metta.cogworks.curriculum.task import Task

from .conftest import create_test_env_config


class TestTask:
    """Test cases for Task class."""

    def test_init(self):
        """Test Task initialization."""
        env_cfg = create_test_env_config()
        task = Task(task_id="test_task_123", env_cfg=env_cfg)

        assert task._task_id == "test_task_123"
        assert task._env_cfg == env_cfg

    def test_env_cfg_property(self):
        """Test env_cfg property access."""
        env_cfg = create_test_env_config()
        task = Task(task_id="test_task", env_cfg=env_cfg)

        assert task.env_cfg == env_cfg

    def test_get_env_config(self):
        """Test get_env_config method."""
        env_cfg = create_test_env_config()
        task = Task(task_id="test_task", env_cfg=env_cfg)

        assert task.get_env_config() == env_cfg

    def test_get_id(self):
        """Test get_id method."""
        env_cfg = create_test_env_config()
        task = Task(task_id="my_unique_task", env_cfg=env_cfg)

        assert task.get_id() == "my_unique_task"

    def test_str_representation(self):
        """Test string representation of Task."""
        env_cfg = create_test_env_config()
        task = Task(task_id="example_task", env_cfg=env_cfg)

        assert str(task) == "Task(id=example_task)"

    def test_with_different_env_configs(self):
        """Test Task with different environment configurations."""
        # Create different GameConfigs to ensure the EnvConfigs are different
        from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, AgentConfig, EnvConfig, GameConfig

        game_config1 = GameConfig(
            num_agents=2,
            agent=AgentConfig(),
            groups={"default": {"id": 0, "props": AgentConfig()}},
            actions=ActionsConfig(noop=ActionConfig()),
            objects={},
        )
        env_cfg1 = EnvConfig(game=game_config1)

        game_config2 = GameConfig(
            num_agents=3,  # Different number of agents
            agent=AgentConfig(),
            groups={"default": {"id": 0, "props": AgentConfig()}},
            actions=ActionsConfig(noop=ActionConfig()),
            objects={},
        )
        env_cfg2 = EnvConfig(game=game_config2)

        task1 = Task(task_id="task1", env_cfg=env_cfg1)
        task2 = Task(task_id="task2", env_cfg=env_cfg2)

        assert task1.env_cfg != task2.env_cfg
        assert task1.get_id() != task2.get_id()

    def test_task_id_types(self):
        """Test Task with different task ID types."""
        env_cfg = create_test_env_config()

        # String task ID
        task_str = Task(task_id="string_id", env_cfg=env_cfg)
        assert task_str.get_id() == "string_id"

        # Numeric string task ID
        task_num_str = Task(task_id="12345", env_cfg=env_cfg)
        assert task_num_str.get_id() == "12345"

        # Empty string task ID
        task_empty = Task(task_id="", env_cfg=env_cfg)
        assert task_empty.get_id() == ""
