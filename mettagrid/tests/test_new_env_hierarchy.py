"""
Test the new environment hierarchy classes.

This test module verifies that our new MettaGrid environment hierarchy
works correctly while being compatible with the existing test framework.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum


def create_test_config():
    """Create a minimal test configuration that works reliably."""
    return DictConfig(
        {
            "game": {
                "max_steps": 20,
                "num_agents": 2,
                "obs_width": 5,
                "obs_height": 5,
                "num_observation_tokens": 25,
                "inventory_item_names": ["heart"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 10,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 0,
                    "rewards": {"heart": 1.0},
                    "action_failure_penalty": 0.0,
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "put_items": {"enabled": True},
                    "get_items": {"enabled": True},
                    "attack": {"enabled": True},
                    "swap": {"enabled": True},
                    "change_color": {"enabled": False},
                    "change_glyph": {"enabled": False, "number_of_glyphs": 0},
                },
                "objects": {
                    "wall": {"type_id": 1, "swappable": False},
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 2,
                    "width": 8,
                    "height": 8,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


@pytest.fixture
def test_curriculum():
    """Create a test curriculum."""
    config = create_test_config()
    return SingleTaskCurriculum("test_hierarchy", config)


class TestNewEnvironmentHierarchy:
    """Test new environment hierarchy classes."""

    def test_imports(self):
        """Test that all new classes can be imported."""
        # Test basic imports - this verifies our modules are structured correctly
        from metta.mettagrid.base_env import MettaGridEnv
        from metta.mettagrid.core import MettaGridCore
        from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
        from metta.mettagrid.puffer_env import MettaGridPufferEnv

        assert MettaGridCore is not None
        assert MettaGridEnv is not None
        assert MettaGridPufferEnv is not None
        assert MettaGridGymEnv is not None
        assert SingleAgentMettaGridGymEnv is not None
        assert MettaGridPettingZooEnv is not None

    def test_puffer_env_creation(self, test_curriculum):
        """Test that MettaGridPufferEnv can be created."""
        from metta.mettagrid.puffer_env import MettaGridPufferEnv

        env = MettaGridPufferEnv(
            curriculum=test_curriculum,
            render_mode=None,
            is_training=False,
        )

        assert env is not None
        env.close()

    def test_puffer_env_basic_ops(self, test_curriculum):
        """Test basic PufferLib environment operations."""
        from metta.mettagrid.puffer_env import MettaGridPufferEnv

        env = MettaGridPufferEnv(
            curriculum=test_curriculum,
            render_mode=None,
            is_training=False,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert obs.shape[0] == env.num_agents

        # Test step
        actions = np.zeros((env.num_agents, 2), dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        assert obs is not None
        assert rewards is not None
        assert terminals is not None
        assert truncations is not None
        assert isinstance(info, dict)

        env.close()

    def test_gym_env_creation(self):
        """Test that Gymnasium environments can be created."""
        from metta.mettagrid.gym_env import SingleAgentMettaGridGymEnv

        # Create single agent config
        config = create_test_config()
        config.game.num_agents = 1
        config.game.map_builder.agents = 1
        curriculum = SingleTaskCurriculum("test_single", config)

        env = SingleAgentMettaGridGymEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=False,
        )

        assert env is not None
        env.close()

    def test_gym_env_basic_ops(self):
        """Test basic Gymnasium environment operations."""
        from metta.mettagrid.gym_env import SingleAgentMettaGridGymEnv

        # Create single agent config
        config = create_test_config()
        config.game.num_agents = 1
        config.game.map_builder.agents = 1
        curriculum = SingleTaskCurriculum("test_single", config)

        env = SingleAgentMettaGridGymEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=False,
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

    def test_pettingzoo_env_creation(self, test_curriculum):
        """Test that PettingZoo environment can be created."""
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        env = MettaGridPettingZooEnv(
            curriculum=test_curriculum,
            render_mode=None,
            is_training=False,
        )

        assert env is not None
        env.close()

    def test_pettingzoo_env_basic_ops(self, test_curriculum):
        """Test basic PettingZoo environment operations."""
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        env = MettaGridPettingZooEnv(
            curriculum=test_curriculum,
            render_mode=None,
            is_training=False,
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

    # Create test curriculum
    config = create_test_config()
    curriculum = SingleTaskCurriculum("manual_test", config)

    # Test imports
    try:
        from metta.mettagrid.puffer_env import MettaGridPufferEnv

        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Test basic creation
    try:
        env = MettaGridPufferEnv(curriculum=curriculum, render_mode=None, is_training=False)
        env.close()
        print("✓ PufferLib environment creation successful")
    except Exception as e:
        print(f"✗ PufferLib environment creation failed: {e}")
        sys.exit(1)

    print("✓ All manual tests passed!")
