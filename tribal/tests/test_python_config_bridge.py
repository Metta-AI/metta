#!/usr/bin/env python3
"""
Test Python-to-Nim configuration bridge integration.

Verifies that Python configuration classes correctly pull defaults from Nim
and that the configuration flow works end-to-end.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add tribal bindings to path
tribal_bindings = Path(__file__).parent.parent / "bindings" / "generated"
if tribal_bindings.exists():
    sys.path.insert(0, str(tribal_bindings))

from experiments.recipes.tribal_basic import make_tribal_environment
from metta.sim.tribal_genny import TribalEnvConfig, TribalGameConfig


class TestPythonConfigBridge(unittest.TestCase):
    """Test the Python-to-Nim configuration bridge."""

    def test_tribal_game_config_from_nim_defaults(self):
        """Test that TribalGameConfig can pull defaults from Nim."""
        config = TribalGameConfig.from_nim_defaults()

        # Check that we got reasonable default values
        self.assertGreater(config.max_steps, 0)
        self.assertGreater(config.ore_per_battery, 0)
        self.assertGreater(config.batteries_per_heart, 0)
        self.assertIsInstance(config.enable_combat, bool)
        self.assertGreaterEqual(config.clippy_spawn_rate, 0.0)
        self.assertLessEqual(config.clippy_spawn_rate, 1.0)
        self.assertGreaterEqual(config.clippy_damage, 0)

        # Reward values should be reasonable
        self.assertGreater(config.heart_reward, 0.0)
        self.assertGreater(config.battery_reward, 0.0)
        self.assertGreater(config.ore_reward, 0.0)
        self.assertLessEqual(config.survival_penalty, 0.0)
        self.assertLessEqual(config.death_penalty, 0.0)

    def test_tribal_env_config_with_nim_defaults(self):
        """Test that TribalEnvConfig can create instances with Nim defaults."""
        config = TribalEnvConfig.with_nim_defaults(label="test_env")

        self.assertEqual(config.label, "test_env")
        self.assertTrue(config.desync_episodes)
        self.assertIsNotNone(config.game)

        # Game config should have valid defaults
        self.assertGreater(config.game.max_steps, 0)
        self.assertIsInstance(config.game.enable_combat, bool)

    def test_recipe_make_tribal_environment(self):
        """Test that the recipe function creates valid configurations."""
        # Test with no overrides (use all Nim defaults)
        config1 = make_tribal_environment()
        self.assertEqual(config1.label, "tribal_basic")
        self.assertGreater(config1.game.max_steps, 0)

        # Test with some overrides
        config2 = make_tribal_environment(max_steps=5000, enable_combat=False)
        self.assertEqual(config2.game.max_steps, 5000)
        self.assertFalse(config2.game.enable_combat)

    def test_consistency_between_creation_methods(self):
        """Test that different creation methods yield consistent defaults."""
        # Create configs using different methods
        direct_game = TribalGameConfig.from_nim_defaults()
        env_config = TribalEnvConfig.with_nim_defaults()
        recipe_config = make_tribal_environment()

        # Check that default values are consistent
        self.assertEqual(direct_game.max_steps, env_config.game.max_steps)
        self.assertEqual(direct_game.max_steps, recipe_config.game.max_steps)

        self.assertEqual(direct_game.heart_reward, env_config.game.heart_reward)
        self.assertEqual(direct_game.heart_reward, recipe_config.game.heart_reward)

        self.assertEqual(direct_game.battery_reward, env_config.game.battery_reward)
        self.assertEqual(direct_game.battery_reward, recipe_config.game.battery_reward)

    def test_environment_creation_from_config(self):
        """Test that environments can be created from Python configs."""
        config = make_tribal_environment()

        # This should not raise an exception
        env = config.create_environment()
        self.assertIsNotNone(env)

        # Basic sanity checks on created environment
        self.assertGreater(env.num_agents, 0)
        self.assertGreater(env.observation_width, 0)
        self.assertGreater(env.observation_height, 0)


class TestConfigurationValues(unittest.TestCase):
    """Test that configuration values are sensible."""

    def setUp(self):
        self.config = TribalGameConfig.from_nim_defaults()

    def test_default_values_are_sensible(self):
        """Test that Nim default values make sense for gameplay."""
        # Time limits
        self.assertGreaterEqual(self.config.max_steps, 1000)  # Should allow meaningful gameplay
        self.assertLessEqual(self.config.max_steps, 10000)  # But not too long

        # Resource chain ratios
        self.assertGreaterEqual(self.config.ore_per_battery, 1)
        self.assertGreaterEqual(self.config.batteries_per_heart, 1)

        # Combat should be balanced
        if self.config.enable_combat:
            self.assertGreater(self.config.clippy_spawn_rate, 0.0)
            self.assertGreater(self.config.clippy_damage, 0)

        # Reward structure should incentivize progress
        self.assertGreater(self.config.heart_reward, self.config.battery_reward)  # Hearts more valuable
        self.assertGreater(self.config.battery_reward, self.config.ore_reward)  # Batteries more valuable than ore


if __name__ == "__main__":
    print("Testing Python-to-Nim configuration bridge...")

    # Suppress gym warnings for cleaner output
    import warnings

    warnings.filterwarnings("ignore", message=".*Gym.*")

    unittest.main(verbosity=2)
