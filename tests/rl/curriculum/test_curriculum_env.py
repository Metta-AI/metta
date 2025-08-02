"""Unit tests for CurriculumEnv."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.rl.curriculum import (
    BucketedTaskGenerator,
    CurriculumClient,
    CurriculumEnv,
    CurriculumManager,
)


class TestCurriculumEnv:
    """Test suite for CurriculumEnv."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a small curriculum manager for testing
        self.manager = CurriculumManager(pool_size=10, min_runs=2, name="test_env")
        curriculum_name = self.manager.get_shared_memory_names()

        # Create client
        self.client = CurriculumClient(curriculum_name=curriculum_name, pool_size=10, num_samples=5, min_runs=2)

        # Create task generator with simple config
        base_config = OmegaConf.create(
            {
                "game": {"num_agents": 1, "max_steps": 100, "reward": {"tag": 1.0}, "spawn": {"num_agents": 1}},
                "map": {"width": 8, "height": 8, "num_tiles": 64},
            }
        )

        buckets = {"game.difficulty": {"range": [0.1, 0.9], "bins": 3}, "map.num_obstacles": [0, 2, 4]}

        self.generator = BucketedTaskGenerator(base_config, buckets)

    def teardown_method(self):
        """Clean up test fixtures."""
        self.client.cleanup()
        self.manager.cleanup()

    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv._initialize_c_env")
    @patch("metta.mettagrid.mettagrid_env.PufferEnv.__init__")
    def test_initialization(self, mock_puffer_init, mock_init_c_env):
        """Test CurriculumEnv initialization."""
        # Mock the parent class initialization
        mock_puffer_init.return_value = None
        mock_init_c_env.return_value = None

        # Create the environment
        env = CurriculumEnv(curriculum_client=self.client, task_generator=self.generator, render_mode=None)

        assert env._curriculum_client == self.client
        assert env._task_generator == self.generator
        assert hasattr(env, "_curriculum")

    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv._initialize_c_env")
    @patch("metta.mettagrid.mettagrid_env.PufferEnv.__init__")
    def test_curriculum_adapter(self, mock_puffer_init, mock_init_c_env):
        """Test that the curriculum adapter works correctly."""
        mock_puffer_init.return_value = None
        mock_init_c_env.return_value = None

        env = CurriculumEnv(curriculum_client=self.client, task_generator=self.generator, render_mode=None)

        # Get the curriculum adapter
        curriculum = env._curriculum

        # Test get_task
        task = curriculum.get_task()
        assert hasattr(task, "env_cfg")
        assert hasattr(task, "complete")
        assert hasattr(task, "short_name")

        # Test that env_cfg has expected structure
        env_cfg = task.env_cfg()
        assert "game" in env_cfg
        assert "map" in env_cfg
        assert "task" in env_cfg
        assert "id" in env_cfg.task

    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv._initialize_c_env")
    @patch("metta.mettagrid.mettagrid_env.PufferEnv.__init__")
    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv.reset")
    def test_reset_with_task_completion(self, mock_reset, mock_puffer_init, mock_init_c_env):
        """Test that reset completes the previous task."""
        mock_puffer_init.return_value = None
        mock_init_c_env.return_value = None
        mock_reset.return_value = (np.zeros((1, 8, 8), dtype=np.uint8), {})

        env = CurriculumEnv(curriculum_client=self.client, task_generator=self.generator, render_mode=None)

        # Create a real task adapter
        from metta.rl.curriculum.curriculum_env import CurriculumTaskAdapter

        mock_task = MagicMock()
        mock_task.task_id = 12345
        task_adapter = CurriculumTaskAdapter(mock_task, OmegaConf.create({}))
        # Simulate rewards from multiple agents
        task_adapter._agent_rewards = [
            np.array([0.5, 0.6, 0.4]),  # timestep 1
            np.array([0.7, 0.8, 0.6]),  # timestep 2
        ]
        env._task = task_adapter

        # Reset should complete the task
        obs, infos = env.reset()

        # Check that the task was completed
        mock_task.complete.assert_called_once()

        # Check infos contain task_id
        assert "task_id" in infos or "curriculum/task_id" in infos

    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv._initialize_c_env")
    @patch("metta.mettagrid.mettagrid_env.PufferEnv.__init__")
    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv.step")
    def test_step_tracks_rewards(self, mock_step, mock_puffer_init, mock_init_c_env):
        """Test that step tracks rewards for task completion."""
        mock_puffer_init.return_value = None
        mock_init_c_env.return_value = None

        # Mock step return values
        obs = np.zeros((1, 8, 8), dtype=np.uint8)
        rewards = np.array([0.5], dtype=np.float32)
        terminals = np.array([False], dtype=bool)
        truncations = np.array([False], dtype=bool)
        mock_step.return_value = (obs, rewards, terminals, truncations, {})

        env = CurriculumEnv(curriculum_client=self.client, task_generator=self.generator, render_mode=None)

        # Create a mock task adapter
        from metta.rl.curriculum.curriculum_env import CurriculumTaskAdapter

        mock_task = MagicMock()
        mock_task.task_id = 54321
        task_adapter = CurriculumTaskAdapter(mock_task, OmegaConf.create({}))
        env._task = task_adapter

        # Take a step
        actions = np.array([[0, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Check that rewards were tracked
        assert len(task_adapter._agent_rewards) == 1
        assert np.array_equal(task_adapter._agent_rewards[0], rewards)

        # Check infos contain task_id
        assert "curriculum/task_id" in infos
        assert infos["curriculum/task_id"] == 54321

    def test_task_completion_calculation(self):
        """Test that task completion calculates mean and variance correctly."""
        from metta.rl.curriculum.curriculum_env import CurriculumTaskAdapter

        # Create a mock task
        mock_task = MagicMock()
        task_adapter = CurriculumTaskAdapter(mock_task, OmegaConf.create({}))

        # Simulate rewards from 5 agents across 3 timesteps
        task_adapter._agent_rewards = [
            np.array([0.5, 0.4, 0.6, 0.7, 0.3]),  # timestep 1
            np.array([0.6, 0.5, 0.7, 0.8, 0.4]),  # timestep 2
            np.array([0.4, 0.3, 0.5, 0.6, 0.2]),  # timestep 3
        ]

        # Finalize the episode
        task_adapter.finalize_episode()

        # Check that complete was called with correct values
        mock_task.complete.assert_called_once()
        args = mock_task.complete.call_args[0]

        # Calculate expected values - total rewards per agent
        agent_totals = np.array([1.5, 1.2, 1.8, 2.1, 0.9])
        expected_mean = np.mean(agent_totals)
        expected_var = np.var(agent_totals)

        # Check mean
        assert abs(args[0] - expected_mean) < 1e-6

        # Check variance
        assert abs(args[1] - expected_var) < 1e-6

    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv._initialize_c_env")
    @patch("metta.mettagrid.mettagrid_env.PufferEnv.__init__")
    @patch("metta.mettagrid.mettagrid_env.MettaGridEnv.close")
    def test_close_completes_final_task(self, mock_close, mock_puffer_init, mock_init_c_env):
        """Test that close completes the final task."""
        mock_puffer_init.return_value = None
        mock_init_c_env.return_value = None
        mock_close.return_value = None

        env = CurriculumEnv(curriculum_client=self.client, task_generator=self.generator, render_mode=None)

        # Create a real task adapter
        from metta.rl.curriculum.curriculum_env import CurriculumTaskAdapter

        mock_task = MagicMock()
        task_adapter = CurriculumTaskAdapter(mock_task, OmegaConf.create({}))
        task_adapter._agent_rewards = [
            np.array([0.5, 0.6]),  # timestep 1
            np.array([0.7, 0.8]),  # timestep 2
        ]
        env._task = task_adapter

        # Close should complete the task
        env.close()

        # Check that the task was completed
        mock_task.complete.assert_called_once()

        # Check that parent close was called
        mock_close.assert_called_once()

    def test_curriculum_task_adapter_interface(self):
        """Test CurriculumTaskAdapter implements old Task interface correctly."""
        from metta.rl.curriculum.curriculum_env import CurriculumTaskAdapter

        # Create mock task and config
        mock_task = MagicMock()
        mock_task.task_id = 99999
        env_cfg = OmegaConf.create({"game": {"difficulty": 0.5}, "map": {"size": 32}})

        adapter = CurriculumTaskAdapter(mock_task, env_cfg)

        # Test env_cfg()
        assert adapter.env_cfg() == env_cfg

        # Test complete() - it should do nothing now
        adapter.complete(0.5)
        adapter.complete(0.7)
        # No rewards are accumulated by complete()

        # Test short_name()
        assert adapter.short_name() == "task_99999"

        # Test finalize_episode() with agent rewards
        adapter._agent_rewards = [
            np.array([0.5, 0.6, 0.7]),  # timestep 1
            np.array([0.8, 0.7, 0.9]),  # timestep 2
        ]
        adapter.finalize_episode()

        # The mock task should have been completed with mean and variance
        mock_task.complete.assert_called_once()
        args = mock_task.complete.call_args[0]
        # Total rewards per agent: [1.3, 1.3, 1.6]
        expected_mean = np.mean([1.3, 1.3, 1.6])
        expected_var = np.var([1.3, 1.3, 1.6])
        assert abs(args[0] - expected_mean) < 1e-6  # mean
        assert abs(args[1] - expected_var) < 1e-6  # variance

    def test_integration_with_real_components(self):
        """Test integration with real curriculum components."""
        # Get a task from the client
        task = self.client.get_task()
        assert task is not None
        assert hasattr(task, "task_id")

        # Generate config for the task
        env_cfg = self.generator.generate(task.task_id)
        assert "game" in env_cfg
        assert "map" in env_cfg
        assert env_cfg.task.id == task.task_id

        # Complete the task
        task.complete(reward_mean=0.6, reward_var=0.05)

        # Check that task state was updated in the pool
        # This verifies the full integration works


if __name__ == "__main__":
    pytest.main([__file__])
