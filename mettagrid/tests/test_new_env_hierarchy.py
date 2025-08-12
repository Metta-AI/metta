"""
Test the new environment hierarchy classes.

This test module verifies that our new MettaGrid environment hierarchy
works correctly while being compatible with the existing test framework.
"""

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.mettagrid_config import EnvConfig, GameConfig
from metta.mettagrid.utils import make_level_map


def _create_env_config(config):
    """Helper function to create EnvConfig from config."""
    from omegaconf import OmegaConf

    # Create a random level map
    level_map = make_level_map(width=10, height=10, num_agents=config.game.num_agents, border_width=1, seed=42)

    # Create game dict and add level_map
    game_dict = OmegaConf.to_container(config.game, resolve=True)
    # Remove the old map config if it exists
    if "map" in game_dict:
        del game_dict["map"]
    # Add the level_map object directly
    game_dict["level_map"] = level_map

    # Create GameConfig and EnvConfig
    game_config = GameConfig(**game_dict)
    return EnvConfig(game=game_config)


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
                    "rewards": {"inventory": {"heart": 1.0}},
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
                "map": {
                    "width": 10,
                    "height": 10,
                    "root": {"type": "metta.map.scenes.random.Random", "params": {"agents": 2}},
                },
            }
        }
    )


# Removed unused curriculum fixture - now using direct env_config creation


class TestNewEnvironmentHierarchy:
    """Test new environment hierarchy classes."""

    def test_imports(self):
        """Test that all new classes can be imported."""
        # Test basic imports - this verifies our modules are structured correctly
        from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
        from metta.mettagrid.mettagrid_env import MettaGridEnv
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        assert MettaGridEnv is not None
        assert MettaGridGymEnv is not None
        assert SingleAgentMettaGridGymEnv is not None
        assert MettaGridPettingZooEnv is not None

    def test_gym_env_creation(self):
        """Test that Gymnasium environments can be created."""
        from metta.mettagrid.gym_env import SingleAgentMettaGridGymEnv

        # Create single agent config
        config = create_test_config()
        config.game.num_agents = 1
        config.game.map.root.params.agents = 1
        env_config = _create_env_config(config)

        env = SingleAgentMettaGridGymEnv(
            env_config=env_config,
            render_mode=None,
        )

        assert env is not None
        env.close()

    def test_gym_env_basic_ops(self):
        """Test basic Gymnasium environment operations."""
        from metta.mettagrid.gym_env import SingleAgentMettaGridGymEnv

        # Create single agent config
        config = create_test_config()
        config.game.num_agents = 1
        config.game.map.root.params.agents = 1
        env_config = _create_env_config(config)

        env = SingleAgentMettaGridGymEnv(
            env_config=env_config,
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

        config = create_test_config()
        env_config = _create_env_config(config)

        env = MettaGridPettingZooEnv(
            env_config=env_config,
            render_mode=None,
        )

        assert env is not None
        env.close()

    def test_pettingzoo_env_basic_ops(self):
        """Test basic PettingZoo environment operations."""
        from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

        config = create_test_config()
        env_config = _create_env_config(config)

        env = MettaGridPettingZooEnv(
            env_config=env_config,
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

    # Create test configuration
    config = create_test_config()
    env_config = _create_env_config(config)

    # Test imports
    try:
        from metta.mettagrid.mettagrid_env import MettaGridEnv

        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Test basic creation
    try:
        env = MettaGridEnv(env_config=env_config, render_mode=None)
        env.close()
        print("✓ MettaGridEnv creation successful")
    except Exception as e:
        print(f"✗ MettaGridEnv creation failed: {e}")
        sys.exit(1)

    print("✓ All manual tests passed!")
