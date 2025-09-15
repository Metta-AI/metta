#!/usr/bin/env python3
"""
Simple test script for tribal PufferLib integration.
"""

import sys
from pathlib import Path
import numpy as np

# Add the tribal src path first
metta_root = Path.cwd()
tribal_src_path = metta_root / 'tribal' / 'src'
pufferlib_path = metta_root / 'PufferLib'
sys.path.insert(0, str(tribal_src_path))
sys.path.insert(0, str(pufferlib_path))

def test_direct_import():
    """Test direct import of tribal environment."""
    print("=== Testing Direct Import ===")
    try:
        from metta.sim.tribal_puffer import TribalPufferEnv
        env = TribalPufferEnv()
        print(f"✓ Environment created: {env.num_agents} agents")

        obs, info = env.reset()
        print(f"✓ Reset successful: {len(obs)} observations")

        # Test step with random actions
        actions = {}
        for agent in env.agents:
            actions[agent] = env.single_action_space.sample()

        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"✓ Step successful: {len(rewards)} rewards")

        return True
    except Exception as e:
        print(f"✗ Direct import failed: {e}")
        return False

def test_pufferlib_import():
    """Test PufferLib factory import."""
    print("\n=== Testing PufferLib Factory ===")
    try:
        from pufferlib.environments.tribal import make
        env = make()
        print(f"✓ Environment created through PufferLib")
        print(f"  Type: {type(env)}")
        print(f"  Single obs space: {env.single_observation_space}")
        print(f"  Single action space: {env.single_action_space}")

        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Obs type: {type(obs)}")

        if isinstance(obs, dict):
            print(f"  Dict obs with keys: {list(obs.keys())}")
            print(f"  First obs shape: {obs[list(obs.keys())[0]].shape}")
        else:
            print(f"  Array obs shape: {obs.shape}")

        return True
    except Exception as e:
        print(f"✗ PufferLib factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    direct_ok = test_direct_import()
    puffer_ok = test_pufferlib_import()

    if direct_ok and puffer_ok:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ Tests failed: Direct={direct_ok}, PufferLib={puffer_ok}")
        sys.exit(1)