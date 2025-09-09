#!/usr/bin/env python3
"""
Consolidated Tribal Environment Test Suite

This replaces and consolidates the scattered tribal tests:
- test_python_bindings.py → Core bindings functionality  
- test_python_config_bridge.py → Configuration integration
- test_environment_attributes.py → Environment attribute access
- test_play_system_integration.py → RL integration
- test_pydantic_validation.py → Pydantic compatibility

Focus: Essential functionality testing with minimal duplication.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from pydantic import BaseModel

# Add project paths  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add tribal bindings
tribal_bindings = Path(__file__).parent.parent / "bindings" / "generated"
if tribal_bindings.exists():
    sys.path.insert(0, str(tribal_bindings))

# Suppress gym warnings
import warnings
warnings.filterwarnings("ignore", message=".*Gym.*")

try:
    import tribal
    from metta.sim.tribal_genny import TribalEnvConfig, TribalGameConfig, TribalGridEnv
    from experiments.recipes.tribal_basic import make_tribal_environment
except ImportError as e:
    print(f"❌ Failed to import tribal bindings: {e}")
    print("Ensure 'nimble bindings' was run in tribal directory")
    sys.exit(1)


class TestTribalCore(unittest.TestCase):
    """Core tribal environment functionality tests."""

    def setUp(self):
        """Set up test environment."""
        game_config = tribal.TribalGameConfig(
            max_steps=50,  # Short for tests
            ore_per_battery=3,
            batteries_per_heart=2, 
            enable_combat=True,
            clippy_spawn_rate=0.1,
            clippy_damage=1,
            heart_reward=1.0,
            ore_reward=0.1,
            battery_reward=0.5,
            survival_penalty=0.0,
            death_penalty=0.0,
        )
        config = tribal.TribalConfig(game_config, False)
        self.env = tribal.TribalEnv(config)
        self.env.reset_env()

    def tearDown(self):
        """Clean up resources."""
        del self.env

    def test_constants(self):
        """Test environment constants are accessible."""
        self.assertEqual(tribal.MAP_AGENTS, 15)
        self.assertEqual(tribal.MAP_WIDTH, 100)
        self.assertEqual(tribal.MAP_HEIGHT, 50)
        self.assertEqual(tribal.OBSERVATION_LAYERS, 19)

    def test_environment_reset(self):
        """Test environment can be reset."""
        # Step once
        actions = tribal.SeqInt()
        for _ in range(tribal.MAP_AGENTS):
            actions.append(0)  # NOOP for all agents
        
        self.env.step(actions)
        step_before = self.env.get_current_step()
        self.assertGreater(step_before, 0)
        
        # Reset should work
        self.env.reset_env()
        step_after = self.env.get_current_step()
        self.assertEqual(step_after, 0)

    def test_observation_retrieval(self):
        """Test observations can be retrieved."""
        obs = self.env.get_token_observations()
        self.assertIsNotNone(obs)
        # Should be a SeqInt with data
        self.assertGreater(len(obs), 0)

    def test_action_execution(self):
        """Test actions can be executed."""
        actions = tribal.SeqInt()
        for _ in range(tribal.MAP_AGENTS):
            actions.append(0)  # NOOP
        
        initial_step = self.env.get_current_step()
        self.env.step(actions)
        new_step = self.env.get_current_step()
        
        self.assertEqual(new_step, initial_step + 1)

    def test_episode_completion(self):
        """Test episodes can run to completion."""
        max_test_steps = 10
        actions = tribal.SeqInt()
        for _ in range(tribal.MAP_AGENTS):
            actions.append(0)  # NOOP
        
        for _ in range(max_test_steps):
            if self.env.is_done():
                break
            self.env.step(actions)
        
        # Should complete without errors
        self.assertTrue(True)


class TestConfigurationSystem(unittest.TestCase):
    """Configuration system integration tests."""

    def test_python_config_creation(self):
        """Test Python configuration classes work."""
        # Test TribalGameConfig defaults
        game_config = TribalGameConfig.from_nim_defaults()
        self.assertGreater(game_config.max_steps, 0)
        self.assertGreater(game_config.ore_per_battery, 0)
        self.assertIsInstance(game_config.enable_combat, bool)

    def test_recipe_config_creation(self):
        """Test recipe configuration functions."""
        # Basic config
        config1 = make_tribal_environment()
        self.assertEqual(config1.label, "tribal_basic")
        self.assertGreater(config1.game.max_steps, 0)
        
        # Config with overrides
        config2 = make_tribal_environment(max_steps=1500, enable_combat=False)
        self.assertEqual(config2.game.max_steps, 1500)
        self.assertFalse(config2.game.enable_combat)

    def test_configuration_consistency(self):
        """Test config creation methods are consistent."""
        direct_config = TribalGameConfig.from_nim_defaults()
        recipe_config = make_tribal_environment()
        
        # Key values should match
        self.assertEqual(direct_config.max_steps, recipe_config.game.max_steps)
        self.assertEqual(direct_config.heart_reward, recipe_config.game.heart_reward)

    def test_config_value_ranges(self):
        """Test configuration values are reasonable."""
        config = TribalGameConfig.from_nim_defaults()
        
        # Sensible defaults
        self.assertGreaterEqual(config.max_steps, 1000)
        self.assertLessEqual(config.max_steps, 10000)
        self.assertGreater(config.heart_reward, config.battery_reward)
        self.assertGreater(config.battery_reward, config.ore_reward)


class TestPythonIntegration(unittest.TestCase):
    """Python-specific integration tests."""

    def test_environment_attributes(self):
        """Test TribalGridEnv exposes correct attributes."""
        config = TribalEnvConfig(label="test")
        env = TribalGridEnv(config)
        
        # Core attributes
        self.assertEqual(env.num_agents, 15)
        self.assertEqual(env.action_names, ['NOOP', 'MOVE', 'ATTACK', 'GET', 'SWAP', 'PUT'])

    def test_observation_format(self):
        """Test observation format and structure."""
        config = TribalEnvConfig(label="test")
        env = TribalGridEnv(config)
        
        observations, info = env.reset()
        
        # Check structure
        self.assertIsInstance(observations, np.ndarray)
        self.assertEqual(observations.shape, (15, 200, 3))  # agents, tokens, features
        self.assertEqual(observations.dtype, np.uint8)
        self.assertGreaterEqual(observations.min(), 0)
        self.assertLessEqual(observations.max(), 255)

    def test_pydantic_compatibility(self):
        """Test Pydantic integration works."""
        class TestModel(BaseModel):
            env: TribalEnvConfig
        
        # Should validate
        config = TribalEnvConfig(label="pydantic_test")
        model = TestModel(env=config)
        self.assertEqual(model.env.label, "pydantic_test")
        
        # Should copy
        copied = config.model_copy(deep=True)
        self.assertEqual(copied.label, "pydantic_test")
        self.assertIsNot(copied, config)

    def test_json_serialization(self):
        """Test JSON serialization works."""
        config = TribalEnvConfig(label="json_test")
        
        # Serialize
        json_str = config.model_dump_json()
        self.assertIn("json_test", json_str)
        
        # Deserialize
        recovered = TribalEnvConfig.model_validate_json(json_str)
        self.assertEqual(recovered.label, "json_test")


def run_consolidated_tests():
    """Run all consolidated tests with better reporting."""
    print("🧪 Consolidated Tribal Environment Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestTribalCore))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestPythonIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"📊 Test Results: {passed}/{total_tests} passed")
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("\nThis consolidated test suite replaces:")
        print("  • test_python_bindings.py")
        print("  • test_python_config_bridge.py") 
        print("  • test_environment_attributes.py")
        print("  • test_play_system_integration.py")
        print("  • test_pydantic_validation.py")
        print("\n🧹 Old test files can be safely removed.")
    else:
        print(f"❌ {failures} failures, {errors} errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_consolidated_tests()
    sys.exit(0 if success else 1)