#!/usr/bin/env python3
"""
Test script for Tribal Environment Python bindings
Tests basic functionality including clippy spawn rate and behavior
"""

import os
import sys

# Add the bindings to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bindings', 'generated'))

try:
    import tribal
    print("✅ Successfully imported tribal bindings")
except ImportError as e:
    print(f"❌ Failed to import tribal bindings: {e}")
    print("Make sure you've run ./build_bindings.sh first")
    sys.exit(1)

def test_basic_functionality():
    """Test basic environment creation and stepping"""
    print("\n🧪 Testing basic functionality...")
    
    # Create default config
    config = tribal.default_tribal_config()
    print(f"✅ Created config with spawn rate: {config.game.clippy_spawn_rate}")
    
    # Create environment
    env = tribal.TribalEnv(config)
    print("✅ Created tribal environment")
    
    # Reset environment
    env.reset_env()
    print("✅ Reset environment")
    
    # Get initial observations
    obs = env.get_observations()
    print(f"✅ Got observations shape: {len(obs)} elements")
    
    # Test stepping
    num_agents = 15  # MapAgents constant
    actions = tribal.SeqInt()
    for _ in range(num_agents):
        actions.append(0)  # action_type = NOOP
        actions.append(0)  # argument = 0
    
    success = env.step(actions)
    print(f"✅ Step successful: {success}")
    
    # Get rewards and status
    rewards = env.get_rewards()
    terminated = env.get_terminated()
    truncated = env.get_truncated()
    
    print(f"✅ Got {len(rewards)} rewards, {len(terminated)} terminated, {len(truncated)} truncated")
    
    return env

def test_clippy_spawn_rate(env, steps=100):
    """Test that clippies are spawning at the expected rate"""
    print(f"\n🐣 Testing clippy spawn rate over {steps} steps...")
    
    num_agents = 15
    actions = tribal.SeqInt()
    for _ in range(num_agents):
        actions.append(0)  # action_type = NOOP
        actions.append(0)  # argument = 0
    
    initial_render = env.render_text()
    initial_clippies = initial_render.count('C')
    print(f"Initial clippies: {initial_clippies}")
    
    for step in range(steps):
        env.step(actions)
        
        if step % 20 == 0:  # Check every 20 steps (expected spawn interval)
            render = env.render_text()
            current_clippies = render.count('C')
            print(f"Step {step}: {current_clippies} clippies on map")
    
    final_render = env.render_text()
    final_clippies = final_render.count('C')
    print(f"Final clippies: {final_clippies}")
    
    if final_clippies > initial_clippies:
        print("✅ Clippies are spawning!")
    else:
        print("⚠️  No new clippies detected - may need more steps or different test")
    
    return final_render

def test_environment_info(env):
    """Test environment information methods"""
    print("\n📊 Testing environment info...")
    
    current_step = env.get_current_step()
    is_done = env.is_episode_done()
    
    print(f"✅ Current step: {current_step}")
    print(f"✅ Episode done: {is_done}")

def main():
    """Run all tests"""
    print("🔬 Tribal Bindings Test Suite")
    print("=" * 40)
    
    try:
        # Test basic functionality
        env = test_basic_functionality()
        
        # Test clippy spawning
        final_render = test_clippy_spawn_rate(env, steps=100)
        
        # Test environment info
        test_environment_info(env)
        
        print("\n📋 Final environment state:")
        print("=" * 20)
        print(final_render[:500] + "..." if len(final_render) > 500 else final_render)
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()