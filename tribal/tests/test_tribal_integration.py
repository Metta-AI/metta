#!/usr/bin/env python3
"""
Consolidated Tribal Environment Integration Tests

This test suite consolidates and replaces the scattered tribal tests with
a comprehensive, well-organized test suite covering:
- Python bindings functionality
- Configuration bridge (Python ↔ Nim)
- Environment creation and attribute access  
- Integration with RL systems
- Pydantic validation

Replaces:
- test_python_bindings.py (basic functionality)
- test_python_config_bridge.py (config integration)
- test_environment_attributes.py (attribute access)
- test_play_system_integration.py (vecenv integration)
- test_pydantic_validation.py (Pydantic compatibility)
"""

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add tribal bindings to path
tribal_bindings = Path(__file__).parent.parent / "bindings" / "generated"
if tribal_bindings.exists():
    sys.path.insert(0, str(tribal_bindings))

# Suppress gym warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", message=".*Gym.*")

try:
    import tribal
    from metta.sim.tribal_genny import TribalEnvConfig, TribalGameConfig, TribalGridEnv
    from experiments.recipes.tribal_basic import make_tribal_environment, tribal_env_curriculum
    from metta.cogworks.curriculum import Curriculum
    from metta.rl.vecenv import make_vecenv
except ImportError as e:
    print(f"❌ Failed to import tribal bindings: {e}")
    print("Make sure to run 'nimble bindings' in the tribal directory first")
    sys.exit(1)


class TestTribalBindings(unittest.TestCase):
    """Test suite for basic tribal environment Python bindings."""

    def setUp(self):
        """Set up test environment with standard config."""
        game_config = tribal.TribalGameConfig(
            max_steps=100,
            ore_per_battery=5,
            batteries_per_heart=3,
            enable_combat=True,
            clippy_spawn_rate=0.01,
            clippy_damage=1,
            heart_reward=1.0,
            ore_reward=0.1,
            battery_reward=0.8,
            survival_penalty=0.0,
            death_penalty=0.0,
        )
        config = tribal.TribalConfig(game_config, False)
        self.env = tribal.TribalEnv(config)
        self.env.reset_env()

    def test_environment_creation(self):
        """Test that tribal environment can be created and initialized."""
        self.assertIsNotNone(self.env)
        self.assertGreater(self.env.num_agents(), 0)

    def test_environment_reset(self):
        """Test environment reset functionality."""
        self.env.reset_env()
        # Environment should be in valid state after reset
        self.assertGreater(self.env.num_agents(), 0)

    def test_observation_retrieval(self):
        """Test that observations can be retrieved."""
        obs = self.env.get_token_observations()
        self.assertIsInstance(obs, list)
        self.assertGreater(len(obs), 0)

    def test_step_execution(self):
        """Test basic environment stepping."""
        initial_obs = self.env.get_token_observations()
        
        # Create dummy actions (NOOP for all agents)
        actions = [0] * self.env.num_agents()
        
        # Step should execute without error
        try:
            self.env.step_env(actions)
        except Exception as e:
            self.fail(f"Environment step failed: {e}")

    def test_episode_completion(self):
        """Test that episodes can run to completion."""
        max_steps = 10  # Short episode for testing
        
        for step in range(max_steps):
            if self.env.env_done():
                break
                
            actions = [0] * self.env.num_agents()  # NOOP actions
            self.env.step_env(actions)
        
        # Should reach end without crashing
        self.assertTrue(True)  # If we get here, test passed


class TestConfigurationBridge(unittest.TestCase):
    """Test Python-to-Nim configuration bridge integration."""

    def test_tribal_game_config_from_nim_defaults(self):
        """Test that TribalGameConfig can pull defaults from Nim."""
        config = TribalGameConfig.from_nim_defaults()

        # Check reasonable default values
        self.assertGreater(config.max_steps, 0)
        self.assertGreater(config.ore_per_battery, 0)
        self.assertGreater(config.batteries_per_heart, 0)
        self.assertIsInstance(config.enable_combat, bool)
        self.assertGreaterEqual(config.clippy_spawn_rate, 0.0)
        self.assertLessEqual(config.clippy_spawn_rate, 1.0)

        # Reward structure validation
        self.assertGreater(config.heart_reward, 0.0)
        self.assertGreater(config.battery_reward, 0.0)
        self.assertGreater(config.ore_reward, 0.0)
        self.assertLessEqual(config.survival_penalty, 0.0)
        self.assertLessEqual(config.death_penalty, 0.0)

    def test_tribal_env_config_with_defaults(self):
        """Test TribalEnvConfig creation with Nim defaults."""
        config = TribalEnvConfig.with_nim_defaults(label="test_env")

        self.assertEqual(config.label, "test_env")
        self.assertTrue(config.desync_episodes)
        self.assertIsNotNone(config.game)
        self.assertGreater(config.game.max_steps, 0)

    def test_recipe_environment_creation(self):
        """Test recipe function creates valid configurations."""
        # Test with default values
        config1 = make_tribal_environment()
        self.assertEqual(config1.label, "tribal_basic")
        self.assertGreater(config1.game.max_steps, 0)

        # Test with custom overrides
        config2 = make_tribal_environment(max_steps=5000, enable_combat=False)
        self.assertEqual(config2.game.max_steps, 5000)
        self.assertFalse(config2.game.enable_combat)

    def test_configuration_consistency(self):
        """Test consistency between different config creation methods."""
        direct_game = TribalGameConfig.from_nim_defaults()
        env_config = TribalEnvConfig.with_nim_defaults()
        recipe_config = make_tribal_environment()

        # Default values should be consistent
        self.assertEqual(direct_game.max_steps, env_config.game.max_steps)
        self.assertEqual(direct_game.max_steps, recipe_config.game.max_steps)
        self.assertEqual(direct_game.heart_reward, env_config.game.heart_reward)
        self.assertEqual(direct_game.battery_reward, recipe_config.game.battery_reward)

    def test_configuration_value_ranges(self):
        """Test that configuration values are in sensible ranges."""
        config = TribalGameConfig.from_nim_defaults()

        # Time limits
        self.assertGreaterEqual(config.max_steps, 1000)
        self.assertLessEqual(config.max_steps, 10000)

        # Resource ratios
        self.assertGreaterEqual(config.ore_per_battery, 1)
        self.assertGreaterEqual(config.batteries_per_heart, 1)

        # Reward progression should make sense
        self.assertGreater(config.heart_reward, config.battery_reward)
        self.assertGreater(config.battery_reward, config.ore_reward)


class TestEnvironmentIntegration(unittest.TestCase):
    """Test integration with RL systems and environment wrappers."""

    def setUp(self):
        """Set up test environment."""
        self.tribal_config = TribalEnvConfig.with_nim_defaults(
            label="test_integration", desync_episodes=True
        )
        self.env = TribalGridEnv(self.tribal_config)

    def test_environment_attributes(self):
        """Test that environment exposes expected attributes."""
        # Core attributes
        self.assertTrue(hasattr(self.env, 'num_agents'))
        self.assertTrue(hasattr(self.env, 'action_names'))
        
        # Verify attribute values
        self.assertGreater(self.env.num_agents, 0)
        self.assertEqual(self.env.num_agents, 15)  # Compile-time constant
        
        expected_actions = ['NOOP', 'MOVE', 'ATTACK', 'GET', 'SWAP', 'PUT']
        self.assertEqual(self.env.action_names, expected_actions)

    def test_observation_structure(self):
        """Test observation format and structure."""
        observations, info = self.env.reset()
        
        self.assertIsInstance(observations, np.ndarray)
        self.assertEqual(len(observations.shape), 3)  # (agents, tokens, features)
        self.assertEqual(observations.shape[0], 15)  # 15 agents
        self.assertEqual(observations.shape[1], 200)  # 200 tokens per agent
        self.assertEqual(observations.shape[2], 3)  # 3 features per token
        self.assertEqual(observations.dtype, np.uint8)

    def test_curriculum_integration(self):
        """Test integration with curriculum system."""
        curriculum_config = tribal_env_curriculum(self.tribal_config)
        curriculum = Curriculum(curriculum_config)
        
        self.assertIsNotNone(curriculum)
        # Curriculum should be able to generate tasks
        task = curriculum.generate_task(0)
        self.assertIsNotNone(task)

    def test_vecenv_integration(self):
        """Test integration with vectorized environment system."""
        curriculum_config = tribal_env_curriculum(self.tribal_config)
        curriculum = Curriculum(curriculum_config)
        
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization="serial",
            num_envs=1,
            render_mode=None,
        )
        
        self.assertIsNotNone(vecenv)
        self.assertEqual(len(vecenv.envs), 1)
        
        # Wrapped environment should maintain key attributes
        wrapped_env = vecenv.envs[0]
        self.assertEqual(wrapped_env.action_names, ['NOOP', 'MOVE', 'ATTACK', 'GET', 'SWAP', 'PUT'])
        self.assertEqual(wrapped_env.num_agents, 15)

    def test_replay_dict_creation(self):
        """Test creating replay dictionaries for visualization."""
        curriculum_config = tribal_env_curriculum(self.tribal_config)
        curriculum = Curriculum(curriculum_config)
        
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization="serial", 
            num_envs=1,
            render_mode=None,
        )
        
        env = vecenv.envs[0]
        
        # Should be able to create replay dict without errors
        replay_dict = {
            "version": 2,
            "action_names": env.action_names,
            "item_names": getattr(env, "resource_names", []),
            "type_names": getattr(env, "object_type_names", []),
            "num_agents": env.num_agents,
            "max_steps": getattr(env, "max_steps", 2000),
            "map_size": [getattr(env, "height", 50), getattr(env, "width", 100)],
            "file_name": "test_replay",
            "steps": [],
        }
        
        self.assertEqual(len(replay_dict), 8)
        self.assertEqual(replay_dict["version"], 2)
        self.assertEqual(replay_dict["num_agents"], 15)


class TestPydanticCompatibility(unittest.TestCase):
    """Test Pydantic integration and validation."""

    class TestModel(BaseModel):
        env: TribalEnvConfig

    def test_pydantic_validation(self):
        """Test that TribalEnvConfig works with Pydantic validation."""
        tribal_config = TribalEnvConfig(label="test")
        
        # Should validate successfully
        test_model = self.TestModel(env=tribal_config)
        self.assertEqual(test_model.env.label, "test")

    def test_model_copy(self):
        """Test Pydantic model_copy functionality."""
        tribal_config = TribalEnvConfig(label="original")
        
        # Deep copy should work
        copied = tribal_config.model_copy(deep=True)
        self.assertEqual(copied.label, "original")
        self.assertIsNot(copied, tribal_config)
        
        # Shallow copy should also work  
        shallow_copied = tribal_config.model_copy(deep=False)
        self.assertEqual(shallow_copied.label, "original")

    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        tribal_config = TribalEnvConfig(label="json_test")
        
        # Should serialize to JSON
        json_str = tribal_config.model_dump_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("json_test", json_str)
        
        # Should deserialize from JSON
        recovered = TribalEnvConfig.model_validate_json(json_str)
        self.assertEqual(recovered.label, "json_test")


class TestTribalEnvironmentSuite(unittest.TestSuite):
    """Custom test suite that runs tests in logical order."""

    def __init__(self):
        super().__init__()
        
        # Add tests in dependency order
        self.addTest(unittest.makeSuite(TestTribalBindings))
        self.addTest(unittest.makeSuite(TestConfigurationBridge))
        self.addTest(unittest.makeSuite(TestEnvironmentIntegration))
        self.addTest(unittest.makeSuite(TestPydanticCompatibility))


def run_tests():
    """Run the consolidated tribal test suite."""
    print("🧪 Running Consolidated Tribal Environment Test Suite")
    print("=" * 60)
    
    suite = TestTribalEnvironmentSuite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tribal tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)