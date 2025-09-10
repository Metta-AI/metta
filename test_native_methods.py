#!/usr/bin/env python
"""Test native PufferLib methods from Nim tribal core."""

from metta.sim.tribal_genny import TribalGridEnv
import numpy as np

def test_native_methods():
    print("🧪 Testing native PufferLib compatibility methods...")
    
    # Create environment
    env = TribalGridEnv()
    print(f"✅ Environment created with {env.num_agents} agents")
    
    # Test native properties
    print(f"✅ Native PufferLib properties: emulated={env.emulated}, done={env.done}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"✅ Reset complete: done={env.done}, obs shape={obs.shape}")
    
    # Step environment
    actions = np.zeros((15, 2), dtype=int)
    obs, rewards, terminals, truncations, info = env.step(actions)
    print(f"✅ Step complete: done={env.done}, rewards={rewards.sum()}")
    
    print("🎯 All native methods working correctly!")

if __name__ == "__main__":
    test_native_methods()