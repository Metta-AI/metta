"""
Test the new environment hierarchy classes.

This test module verifies that our new MettaGrid environment hierarchy
works correctly while being compatible with the existing test framework.
"""

import numpy as np

from metta.mettagrid.config.builder import make_arena


class TestNewEnvironmentHierarchy:
    """Test new environment hierarchy classes."""

    def test_imports(self):
        """Test that all new classes can be imported."""
        # Test basic imports - this verifies our modules are structured correctly
        from metta.mettagrid.gym_env import MettaGridGymEnv
        from metta.mettagrid.mettagrid_env import MettaGridEnv
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        assert MettaGridEnv is not None
        assert MettaGridGymEnv is not None
        assert MettaGridPettingZooEnv is not None

    def test_gym_env_creation(self):
        """Test that Gymnasium environments can be created."""
        from metta.mettagrid.gym_env import MettaGridGymEnv

        env = MettaGridGymEnv(
            make_arena(num_agents=1),
            render_mode=None,
        )

        assert env is not None
        env.close()

    def test_gym_env_basic_ops(self):
        """Test basic Gymnasium environment operations."""
        from metta.mettagrid.gym_env import MettaGridGymEnv

        env = MettaGridGymEnv(
            make_arena(num_agents=1),
            render_mode=None,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        assert obs is not None

        # Test step
        action = np.array([0, 0], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_pettingzoo_env_creation(self):
        """Test that PettingZoo environment can be created."""
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        # Create PettingZoo config
        env = MettaGridPettingZooEnv(
            make_arena(num_agents=3),
            render_mode=None,
        )

        assert env is not None
        env.close()

    def test_pettingzoo_env_basic_ops(self):
        """Test basic PettingZoo environment operations."""
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        # Create multi-agent config
        env = MettaGridPettingZooEnv(
            make_arena(num_agents=3),
            render_mode=None,
        )

        # Test reset
        observations, infos = env.reset(seed=42)
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        assert len(observations) > 0

        # Test step
        actions = {agent: np.array([0, 0], dtype=np.int32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        env.close()


if __name__ == "__main__":
    # Run tests manually if called directly
    import sys

    print("Running manual tests...")

    # Test imports
    try:
        from metta.mettagrid.mettagrid_env import MettaGridEnv

        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Test basic creation
    try:
        env = MettaGridEnv(make_arena(num_agents=24), render_mode=None)
        env.close()
        print("✓ MettaGridEnv creation successful")
    except Exception as e:
        print(f"✗ MettaGridEnv creation failed: {e}")
        sys.exit(1)

    print("✓ All manual tests passed!")
