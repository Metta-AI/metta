"""Tests for MettaGridPufferEnv - PufferLib adapter."""

import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator import Simulator
from pufferlib.emulation import GymnasiumPufferEnv


@pytest.fixture
def puffer_sim_config():
    """Create a basic simulation config for Puffer testing."""
    return MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            obs=ObsConfig(width=3, height=3, num_tokens=50),
            max_steps=10,
            resource_names=["ore", "wood"],
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig()},
            map_builder=RandomMapBuilder.Config(width=7, height=5, agents=2, seed=42),
        )
    )


@pytest.fixture
def simulator():
    """Create a simulator instance."""
    sim = Simulator()
    yield sim
    # Clean up after test
    sim.close()


class TestMettaGridPufferEnvCreation:
    """Test environment creation and initialization."""

    def test_env_creation(self, simulator, puffer_sim_config):
        """Test that PufferEnv can be created successfully."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        assert env is not None
        assert env.num_agents == 2
        assert env.single_observation_space is not None
        assert env.single_action_space is not None

    def test_observation_space(self, simulator, puffer_sim_config):
        """Test observation space properties."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        obs_space = env.single_observation_space
        assert obs_space.shape == (50, 3)  # num_observation_tokens x 3
        assert obs_space.dtype == np.uint8

    def test_action_space(self, simulator, puffer_sim_config):
        """Test action space properties."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        action_space = env.single_action_space
        assert action_space.n > 0
        # Should have at least noop and move actions
        assert action_space.n >= 5  # noop + 4 cardinal directions

    def test_metadata_present(self, simulator, puffer_sim_config):
        """Ensure Gymnasium vector wrappers can read env metadata."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        assert isinstance(env.metadata, dict)
        assert "render_modes" in env.metadata
        assert "ansi" in env.metadata["render_modes"]


class TestMettaGridPufferEnvGymnasiumVector:
    """Tests for Gymnasium vector compatibility."""

    def test_async_vector_env_constructs(self, puffer_sim_config):
        """Gymnasium AsyncVectorEnv should reset/step without metadata errors."""

        def make_env():
            sim = Simulator()
            return GymnasiumPufferEnv(
                env_creator=MettaGridPufferEnv,
                env_kwargs={"simulator": sim, "cfg": puffer_sim_config},
            )

        vec = AsyncVectorEnv([make_env])
        obs, info = vec.reset()
        assert obs.shape[0] == 1

        actions = np.zeros((1, puffer_sim_config.game.num_agents), dtype=np.int32)
        obs, reward, term, trunc, info = vec.step(actions)
        assert reward.shape == (1,)
        vec.close()


class TestMettaGridPufferEnvReset:
    """Test reset functionality."""

    def test_reset_returns_observations(self, simulator, puffer_sim_config):
        """Test that reset returns observations."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        obs, info = env.reset()

        assert obs is not None
        assert obs.shape == (2, 50, 3)  # num_agents x num_tokens x 3
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

    def test_reset_with_seed(self, simulator, puffer_sim_config):
        """Test that reset accepts a seed parameter."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # With same seed, initial observations should be identical
        np.testing.assert_array_equal(obs1, obs2)

    def test_multiple_resets(self, simulator, puffer_sim_config):
        """Test that environment can be reset multiple times."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        for _ in range(3):
            obs, info = env.reset()
            assert obs is not None
            assert obs.shape == (2, 50, 3)


class TestMettaGridPufferEnvStep:
    """Test step functionality."""

    def test_step_returns_correct_types(self, simulator, puffer_sim_config):
        """Test that step returns correctly typed outputs."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)
        env.reset()

        # Create noop actions for all agents
        actions = np.zeros(2, dtype=np.int32)

        obs, rewards, terminals, truncations, info = env.step(actions)

        assert obs.shape == (2, 50, 3)
        assert obs.dtype == np.uint8
        assert rewards.shape == (2,)
        assert rewards.dtype == np.float32
        assert terminals.shape == (2,)
        assert terminals.dtype == bool
        assert truncations.shape == (2,)
        assert truncations.dtype == bool
        assert isinstance(info, dict)

    def test_step_with_actions(self, simulator, puffer_sim_config):
        """Test stepping with different actions."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)
        env.reset()

        # Get action indices
        action_names = env._sim.action_names
        noop_idx = action_names.index("noop")

        # Step with noop for both agents
        actions = np.array([noop_idx, noop_idx], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        assert not terminals.any(), "Agents should not be terminated after one noop"
        assert not truncations.any(), "Agents should not be truncated after one step"

    def test_episode_completion(self, simulator):
        """Test that episode completes at max_steps."""
        config = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                obs=ObsConfig(width=3, height=3, num_tokens=20),
                max_steps=3,  # Short episode
                episode_truncates=True,  # Enable truncation at max_steps
                actions=ActionsConfig(noop=NoopActionConfig()),
                map_builder=RandomMapBuilder.Config(width=5, height=5, agents=1, seed=42),
            )
        )

        env = MettaGridPufferEnv(simulator, config)
        env.reset()

        noop_idx = env._sim.action_names.index("noop")
        actions = np.array([noop_idx], dtype=np.int32)

        # Take 3 steps
        for _ in range(3):
            obs, rewards, terminals, truncations, info = env.step(actions)

        # Should be truncated at max_steps
        assert truncations[0], "Episode should be truncated after max_steps"

    def test_auto_reset_on_episode_end(self, simulator):
        """Test that environment automatically resets when all agents are done."""
        config = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                obs=ObsConfig(width=3, height=3, num_tokens=20),
                max_steps=2,  # Very short episode
                episode_truncates=True,  # Enable truncation at max_steps
                actions=ActionsConfig(noop=NoopActionConfig()),
                map_builder=RandomMapBuilder.Config(width=5, height=5, agents=2, seed=42),
            )
        )

        env = MettaGridPufferEnv(simulator, config)
        env.reset()

        noop_idx = env._sim.action_names.index("noop")
        actions = np.array([noop_idx, noop_idx], dtype=np.int32)

        # Store initial simulation reference
        initial_sim = env.current_simulation

        # Step until episode ends (all agents truncated)
        for _ in range(2):
            obs, rewards, terminals, truncations, info = env.step(actions)

        # Verify all agents are truncated
        assert truncations.all(), "All agents should be truncated after max_steps"

        # Take another step - this should auto-reset before stepping
        obs_after_reset, rewards_after_reset, terminals_after_reset, truncations_after_reset, info_after_reset = (
            env.step(actions)
        )

        # After auto-reset, should have valid observations
        assert obs_after_reset.shape == (2, 20, 3), "Observations should be valid after auto-reset"
        assert obs_after_reset.dtype == np.uint8, "Observations should have correct dtype"

        # Should have a new simulation instance (auto-reset happened)
        assert env.current_simulation is not initial_sim, "Auto-reset should create new simulation"

        # Should be able to continue stepping after auto-reset
        obs_next, rewards_next, terminals_next, truncations_next, info_next = env.step(actions)
        assert obs_next.shape == (2, 20, 3), "Should be able to continue stepping after auto-reset"


class TestMettaGridPufferEnvBuffers:
    """Test buffer management."""

    def test_buffer_properties_accessible(self, simulator, puffer_sim_config):
        """Test that buffer properties can be accessed."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        assert env.observations is not None
        assert env.rewards is not None
        assert env.terminals is not None
        assert env.truncations is not None
        assert env.masks is not None
        assert env.actions is not None

    def test_buffer_shapes(self, simulator, puffer_sim_config):
        """Test that buffers have correct shapes."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        assert env.observations.shape == (2, 50, 3)
        assert env.rewards.shape == (2,)
        assert env.terminals.shape == (2,)
        assert env.truncations.shape == (2,)
        assert env.masks.shape == (2,)
        assert env.actions.shape == (2,)

    def test_buffers_persist_across_resets(self, simulator, puffer_sim_config):
        """Test that buffer objects persist across resets."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        # Get buffer object references
        obs_buf_id = id(env.observations)
        rewards_buf_id = id(env.rewards)

        env.reset()

        # Buffer objects should be the same after reset
        assert id(env.observations) == obs_buf_id
        assert id(env.rewards) == rewards_buf_id


class TestMettaGridPufferEnvIntegration:
    """Test integration scenarios."""

    def test_full_episode_workflow(self, simulator, puffer_sim_config):
        """Test complete episode from reset to completion."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)

        # Reset
        obs, info = env.reset()
        assert obs.shape == (2, 50, 3)

        # Run a few steps
        noop_idx = env._sim.action_names.index("noop")
        actions = np.array([noop_idx, noop_idx], dtype=np.int32)

        for _ in range(5):
            obs, rewards, terminals, truncations, info = env.step(actions)

            if terminals.any() or truncations.any():
                break

        # Should be able to reset again
        obs, info = env.reset()
        assert obs.shape == (2, 50, 3)

    def test_close(self, simulator, puffer_sim_config):
        """Test that environment can be closed."""
        env = MettaGridPufferEnv(simulator, puffer_sim_config)
        env.reset()

        # Close should not raise
        env.close()


class TestMettaGridPufferEnvSupervisorPolicy:
    """Test supervisor policy functionality."""

    def test_mettagrid_puffer_env_supervisor_policy(self, simulator, puffer_sim_config):
        """Test that supervisor policy correctly sets teacher_actions."""

        # Create supervisor policy spec
        supervisor_policy_spec = PolicySpec(class_path="noop")

        # Create environment with supervisor
        env = MettaGridPufferEnv(simulator, puffer_sim_config, supervisor_policy_spec=supervisor_policy_spec)

        # Verify supervisor policy was initialized
        assert env._env_supervisor is not None

        # Reset environment
        obs, info = env.reset()

        # Get action names from simulation
        action_names = env._sim.action_names

        # Verify teacher_actions are set after reset
        teacher_actions = env.teacher_actions
        assert teacher_actions.shape == (2,)
        # Check that dtype is an integer type (int32 or int64 depending on platform)
        assert np.issubdtype(teacher_actions.dtype, np.integer)

        # NoopPolicy should always return noop actions
        noop_idx = action_names.index("noop")
        assert teacher_actions[0] == noop_idx
        assert teacher_actions[1] == noop_idx

        # Step the environment
        actions = np.array([noop_idx, noop_idx], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # After step, teacher_actions should still be noop (NoopPolicy always returns noop)
        teacher_actions_after_step = env.teacher_actions
        assert teacher_actions_after_step[0] == noop_idx
        assert teacher_actions_after_step[1] == noop_idx

        # Step again - should still be noop
        obs, rewards, terminals, truncations, info = env.step(actions)
        teacher_actions_after_step2 = env.teacher_actions
        assert teacher_actions_after_step2[0] == noop_idx
        assert teacher_actions_after_step2[1] == noop_idx

        # Step again - should still be noop
        obs, rewards, terminals, truncations, info = env.step(actions)
        teacher_actions_after_step3 = env.teacher_actions
        assert teacher_actions_after_step3[0] == noop_idx
        assert teacher_actions_after_step3[1] == noop_idx

        # Step again - should still be noop
        obs, rewards, terminals, truncations, info = env.step(actions)
        teacher_actions_after_step4 = env.teacher_actions
        assert teacher_actions_after_step4[0] == noop_idx
        assert teacher_actions_after_step4[1] == noop_idx
