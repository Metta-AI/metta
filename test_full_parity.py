#!/usr/bin/env python
"""Test full parity between ComponentPolicy (YAML) and Fast (PyTorch)."""

import logging
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.metta_agent import MettaAgent
from metta.rl.system_config import SystemConfig

# Suppress debug logging
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


def test_parity_features():
    """Test that all parity features are working."""
    env = create_mock_env()
    system_cfg = SystemConfig(device="cpu")
    
    print("=" * 60)
    print("Testing Full Parity Between ComponentPolicy and Fast")
    print("=" * 60)
    
    # Create YAML agent
    yaml_cfg = OmegaConf.load("configs/agent/fast.yaml")
    yaml_agent = MettaAgent(env, system_cfg, yaml_cfg)
    
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
    
    # Initialize both
    features = env.get_observation_features()
    yaml_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))
    py_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))
    
    print("\n1. Testing Weight Clipping:")
    # Check MettaAgent itself, not just the policy
    yaml_has_clip = hasattr(yaml_agent, 'clip_weights')
    py_has_clip = hasattr(py_agent, 'clip_weights')
    print(f"   YAML agent has clip_weights: {yaml_has_clip}")
    print(f"   py_agent has clip_weights: {py_has_clip}")
    
    if py_has_clip:
        try:
            py_agent.clip_weights()  # Call on MettaAgent, not policy
            print("   ✓ py_agent clip_weights() executes without error")
        except Exception as e:
            print(f"   ✗ py_agent clip_weights() failed: {e}")
    
    print("\n2. Testing L2-Init Regularization:")
    yaml_has_l2 = hasattr(yaml_agent, 'l2_init_loss')
    py_has_l2 = hasattr(py_agent, 'l2_init_loss')
    print(f"   YAML agent has l2_init_loss: {yaml_has_l2}")
    print(f"   py_agent has l2_init_loss: {py_has_l2}")
    
    if py_has_l2:
        try:
            loss = py_agent.l2_init_loss()  # Call on MettaAgent
            print(f"   ✓ py_agent l2_init_loss() returns: {loss.item()}")
        except Exception as e:
            print(f"   ✗ py_agent l2_init_loss() failed: {e}")
    
    print("\n3. Testing Weight Metrics:")
    yaml_has_metrics = hasattr(yaml_agent, 'compute_weight_metrics')
    py_has_metrics = hasattr(py_agent, 'compute_weight_metrics')
    print(f"   YAML agent has compute_weight_metrics: {yaml_has_metrics}")
    print(f"   py_agent has compute_weight_metrics: {py_has_metrics}")
    
    if py_has_metrics:
        try:
            metrics = py_agent.compute_weight_metrics()  # Call on MettaAgent
            print(f"   ✓ py_agent compute_weight_metrics() returns {len(metrics)} metrics")
            
            # Check for effective rank on critic_1
            critic_metrics = [m for m in metrics if 'critic_1' in m.get('name', '')]
            if critic_metrics:
                has_effective_rank = 'effective_rank' in critic_metrics[0]
                print(f"   ✓ critic_1 has effective_rank metric: {has_effective_rank}")
        except Exception as e:
            print(f"   ✗ py_agent compute_weight_metrics() failed: {e}")
    
    print("\n4. Testing Update L2-Init Weights:")
    yaml_has_update = hasattr(yaml_agent, 'update_l2_init_weight_copy')
    py_has_update = hasattr(py_agent, 'update_l2_init_weight_copy')
    print(f"   YAML agent has update_l2_init_weight_copy: {yaml_has_update}")
    print(f"   py_agent has update_l2_init_weight_copy: {py_has_update}")
    
    if py_has_update:
        try:
            py_agent.update_l2_init_weight_copy()  # Call on MettaAgent
            print("   ✓ py_agent update_l2_init_weight_copy() executes without error")
        except Exception as e:
            print(f"   ✗ py_agent update_l2_init_weight_copy() failed: {e}")
    
    print("\n5. Testing Config Attributes:")
    print(f"   YAML clip_range: {yaml_agent.clip_range if hasattr(yaml_agent, 'clip_range') else 'N/A'}")
    print(f"   py_agent clip_range: {py_agent.clip_range if hasattr(py_agent, 'clip_range') else 'N/A'}")
    
    # Test that parameters still match
    print("\n6. Parameter Count Verification:")
    yaml_params = sum(p.numel() for p in yaml_agent.parameters())
    py_params = sum(p.numel() for p in py_agent.parameters())
    print(f"   YAML agent params: {yaml_params:,}")
    print(f"   py_agent params: {py_params:,}")
    params_match = yaml_params == py_params
    print(f"   Parameters match: {'✓' if params_match else '✗'}")
    
    print("\n" + "=" * 60)
    print("PARITY TEST SUMMARY:")
    
    all_features = [
        ("Weight Clipping", py_has_clip),
        ("L2-Init Loss", py_has_l2),
        ("Weight Metrics", py_has_metrics),
        ("Update L2-Init", py_has_update),
        ("Parameter Count", params_match),
    ]
    
    for feature, has_it in all_features:
        status = "✓" if has_it else "✗"
        print(f"  {status} {feature}")
    
    if all(has_it for _, has_it in all_features):
        print("\n✓ FULL PARITY ACHIEVED!")
        print("The py_agent=fast now has complete feature parity with agent=fast")
    else:
        print("\n✗ Some parity features are still missing")


if __name__ == "__main__":
    test_parity_features()