#!/usr/bin/env python3
"""
Test script to verify tribal PufferLib integration.

This script tests that the tribal environment can be created with the
PufferLib wrapper and used with the training infrastructure.
"""

import numpy as np
import sys
from pathlib import Path

def test_tribal_puffer_import():
    """Test that we can import the tribal PufferLib wrapper."""
    print("Testing tribal PufferLib import...")
    try:
        from metta.sim.tribal_puffer import TribalPufferEnv, make_tribal_puffer_env
        print("✅ Successfully imported TribalPufferEnv")
        return True
    except ImportError as e:
        print(f"❌ Failed to import TribalPufferEnv: {e}")
        return False

def test_tribal_puffer_creation():
    """Test creating a tribal PufferLib environment."""
    print("Testing tribal PufferLib environment creation...")
    try:
        from metta.sim.tribal_puffer import make_tribal_puffer_env
        
        config = {
            "max_steps": 100,
            "heart_reward": 1.0,
            "battery_reward": 0.8,
        }
        
        env = make_tribal_puffer_env(**config)
        print("✅ Successfully created TribalPufferEnv")
        
        # Check PufferLib properties
        print(f"   - Single observation space: {env.single_observation_space}")
        print(f"   - Single action space: {env.single_action_space}")
        print(f"   - Emulated: {env.emulated}")
        print(f"   - Number of agents: {env.num_agents}")
        
        return True, env
    except Exception as e:
        print(f"❌ Failed to create TribalPufferEnv: {e}")
        return False, None

def test_tribal_puffer_reset_step():
    """Test reset and step operations."""
    print("Testing tribal PufferLib reset and step...")
    try:
        from metta.sim.tribal_puffer import make_tribal_puffer_env
        
        env = make_tribal_puffer_env(max_steps=50)
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful. Obs shape: {obs.shape}, Info: {info}")
        
        # Test step with random actions
        num_agents = env.num_agents
        actions = np.random.randint(0, [6, 8], size=(num_agents, 2), dtype=np.int32)
        
        obs, rewards, terminals, truncations, info = env.step(actions)
        print(f"✅ Step successful. Obs shape: {obs.shape}, Rewards: {rewards}, Terminals: {terminals}")
        
        # Test multiple steps
        for i in range(5):
            actions = np.random.randint(0, [6, 8], size=(num_agents, 2), dtype=np.int32)
            obs, rewards, terminals, truncations, info = env.step(actions)
            if terminals.any() or truncations.any():
                print(f"   Episode ended at step {i+1}")
                break
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed reset/step test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pufferlib_async_interface():
    """Test the PufferLib async interface."""
    print("Testing PufferLib async interface...")
    try:
        from metta.sim.tribal_puffer import make_tribal_puffer_env
        
        env = make_tribal_puffer_env(max_steps=50)
        
        # Test async_reset
        obs = env.async_reset()
        print(f"✅ Async reset successful. Obs shape: {obs.shape}")
        
        # Test recv after reset
        result = env.recv()
        print(f"✅ Recv after reset successful. Got {len(result)} items")
        
        # Test send/recv cycle
        num_agents = env.num_agents
        actions = np.random.randint(0, [6, 8], size=(num_agents, 2), dtype=np.int32)
        
        env.send(actions)
        result = env.recv()
        obs, rewards, terminals, truncations, info_list, lives, scores = result
        print(f"✅ Send/recv cycle successful. Obs shape: {obs.shape}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed async interface test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vecenv_integration():
    """Test integration with vecenv."""
    print("Testing vecenv integration...")
    try:
        from metta.cogworks.curriculum import TaskConfig, Curriculum
        from metta.sim.tribal_genny import TribalEnvConfig
        from metta.rl.vecenv import make_env_func
        
        # Create a simple tribal curriculum
        tribal_config = TribalEnvConfig(
            game={"max_steps": 100}
        )
        task_config = TaskConfig(env_cfg=tribal_config)
        curriculum = Curriculum(tasks=[task_config])
        
        # Create environment through make_env_func
        env = make_env_func(curriculum)
        print("✅ Vecenv integration successful")
        
        # Test that we got the right type
        print(f"   Environment type: {type(env)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed vecenv integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Tribal PufferLib Integration")
    print("=" * 50)
    
    tests = [
        test_tribal_puffer_import,
        test_tribal_puffer_creation,
        test_tribal_puffer_reset_step,
        test_pufferlib_async_interface,
        test_vecenv_integration,
    ]
    
    passed = 0
    for test in tests:
        try:
            result = test()
            if result is True or (isinstance(result, tuple) and result[0] is True):
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Tribal PufferLib integration is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the tribal bindings and environment setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())