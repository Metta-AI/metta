#!/usr/bin/env python
"""Test the Fast policy forward pass in training mode."""

import logging
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent
from metta.rl.system_config import SystemConfig

# Set up logging
logging.basicConfig(level=logging.WARNING)


def create_mock_env():
    """Create a mock environment for testing."""
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=255, shape=(200, 3), dtype=np.uint8),
        single_action_space=gym.spaces.MultiDiscrete([9, 1, 2, 4, 1, 1, 1]),
        obs_width=11,
        obs_height=11,
        feature_normalizations={i: 1.0 for i in range(25)},
        action_names=["attack", "get_items", "move", "noop", "put_items", "rotate", "swap"],
        max_action_args=[8, 0, 1, 3, 0, 0, 0],
    )

    def get_observation_features():
        return {f"feature_{i}": {"id": i, "normalization": 1.0} for i in range(25)}

    env.get_observation_features = get_observation_features
    return env


def test_forward_pass():
    """Test forward pass in training mode for both agents."""
    env = create_mock_env()
    system_cfg = SystemConfig(device="cpu")
    
    # Create py_agent
    py_agent_cfg = OmegaConf.create(
        {
            "agent_type": "fast",
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "observations": {"obs_key": "grid_obs"},
        }
    )
    py_agent = MettaAgent(env, system_cfg, py_agent_cfg)
    
    # Initialize
    features = env.get_observation_features()
    py_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))
    
    # Create test data with BPTT dimension
    B = 4  # batch size
    T = 8  # time steps
    
    # Create observations
    test_obs = torch.randint(0, 255, (B, T, 200, 3), dtype=torch.uint8)
    
    # Create actions (action_type, action_param)
    test_actions = torch.zeros((B, T, 2), dtype=torch.long)
    # Add some random valid actions
    test_actions[:, :, 0] = torch.randint(0, 7, (B, T))  # action types
    test_actions[:, :, 1] = 0  # action params (simplified)
    
    # Create TensorDict
    td = TensorDict(
        {
            "env_obs": test_obs,
        },
        batch_size=(B, T),
    )
    
    print("Testing py_agent forward pass with BPTT...")
    print(f"Input shapes:")
    print(f"  env_obs: {test_obs.shape}")
    print(f"  actions: {test_actions.shape}")
    
    try:
        output = py_agent(td, action=test_actions)
        print("\nForward pass successful!")
        print(f"Output keys: {list(output.keys())}")
        
        for key in ["act_log_prob", "entropy", "value", "full_log_probs"]:
            if key in output:
                print(f"  {key}: shape={output[key].shape}")
        
        # Check expected shapes
        assert output["act_log_prob"].shape == (B, T), f"act_log_prob shape mismatch: {output['act_log_prob'].shape}"
        assert output["entropy"].shape == (B, T), f"entropy shape mismatch: {output['entropy'].shape}"
        assert output["value"].shape == (B, T, 1), f"value shape mismatch: {output['value'].shape}"
        assert output["full_log_probs"].shape[0:2] == (B, T), f"full_log_probs shape mismatch: {output['full_log_probs'].shape}"
        
        print("\nAll shape checks passed!")
        
    except Exception as e:
        print(f"\nForward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_yaml_agent():
    """Test YAML agent for comparison."""
    env = create_mock_env()
    system_cfg = SystemConfig(device="cpu")
    
    # Create YAML agent
    yaml_cfg = OmegaConf.load("configs/agent/fast.yaml")
    yaml_agent = MettaAgent(env, system_cfg, yaml_cfg)
    
    # Initialize
    features = env.get_observation_features()
    yaml_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))
    
    # Create test data
    B = 4
    T = 8
    test_obs = torch.randint(0, 255, (B, T, 200, 3), dtype=torch.uint8)
    test_actions = torch.zeros((B, T, 2), dtype=torch.long)
    test_actions[:, :, 0] = torch.randint(0, 7, (B, T))
    
    td = TensorDict(
        {"env_obs": test_obs},
        batch_size=(B, T),
    )
    
    print("\nTesting YAML agent forward pass with BPTT...")
    
    try:
        output = yaml_agent(td, action=test_actions)
        print("YAML Forward pass successful!")
        print(f"Output shapes:")
        for key in ["act_log_prob", "entropy", "value", "full_log_probs"]:
            if key in output:
                print(f"  {key}: {output[key].shape}")
        return True
    except Exception as e:
        print(f"YAML Forward pass failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fast Policy Forward Pass Fix")
    print("=" * 60)
    
    py_success = test_forward_pass()
    yaml_success = test_yaml_agent()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  py_agent (Fast): {'✓ PASSED' if py_success else '✗ FAILED'}")
    print(f"  YAML agent: {'✓ PASSED' if yaml_success else '✗ FAILED'}")
    
    if py_success and yaml_success:
        print("\n✓ Both agents work correctly with BPTT!")
    else:
        print("\n✗ There are still issues to fix")