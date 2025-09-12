#!/usr/bin/env -S uv run python
"""Quick test script for SmolLM2 agent."""

import torch
from metta.agent.agent_config import AgentConfig, create_agent
import numpy as np
import gymnasium as gym

# Create a mock environment for testing
class MockEnv:
    def __init__(self):
        self.single_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(200, 3), dtype=np.uint8
        )
        self.single_action_space = gym.spaces.MultiDiscrete([4, 4])
        self.obs_width = 11
        self.obs_height = 11
        self.feature_normalizations = {}
        self.max_action_args = [3, 3]  # Example: 4 actions each with max 3 parameters

# Test agent creation
print("Testing SmolLM2 agent creation...")
    
try:
    # Create mock environment
    env = MockEnv()
    
    # Configure SmolLM2 agent
    agent_cfg = AgentConfig(
        name="pytorch/smollm2",
        clip_range=0.2,
        analyze_weights_interval=500,
    )
    
    # Create agent
    print("Creating SmolLM2 agent...")
    agent = create_agent(
        config=agent_cfg,
        env=env,
    )
    
    print(f"✓ Agent created successfully!")
    print(f"  - Type: {type(agent).__name__}")
    print(f"  - Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    batch_size = 2
    obs = torch.randint(0, 255, (batch_size, 200, 3), dtype=torch.uint8)
    
    # Create a TensorDict for the forward pass
    from tensordict import TensorDict
    td = TensorDict({
        "env_obs": obs,
    }, batch_size=[batch_size])
    
    # Run forward pass
    with torch.no_grad():
        output = agent(td)
    
    print(f"✓ Forward pass successful!")
    print(f"  - Output keys: {list(output.keys())}")
    if "actions" in output:
        print(f"  - Action shape: {output['actions'].shape}")
    if "values" in output:
        print(f"  - Value shape: {output['values'].shape}")
    
    print("\n✅ All tests passed! SmolLM2 agent is working.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()