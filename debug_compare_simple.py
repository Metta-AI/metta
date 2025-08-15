#!/usr/bin/env python
"""Simple debug script to compare YAML agent=fast vs py_agent=fast"""

import logging
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.metta_agent import MettaAgent
from metta.rl.system_config import SystemConfig

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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

    # Add method to get observation features
    def get_observation_features():
        return {f"feature_{i}": {"id": i, "normalization": 1.0} for i in range(25)}

    env.get_observation_features = get_observation_features

    return env


def analyze_agent(agent, name):
    """Analyze and print agent structure."""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {name}")
    print(f"{'=' * 60}")

    # Overall stats
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Policy type
    print(f"\nPolicy type: {type(agent.policy).__name__}")
    print(f"Policy class: {agent.policy.__class__}")

    # Check for key attributes
    if hasattr(agent.policy, "components"):
        print("\nComponentPolicy components:")
        for name, comp in agent.policy.components.items():
            param_count = sum(p.numel() for p in comp.parameters()) if hasattr(comp, "parameters") else 0
            print(f"  {name}: {comp.__class__.__name__} ({param_count:,} params)")

    # Check specific components for py_agent Fast
    if hasattr(agent.policy, "policy"):
        policy = agent.policy.policy
        print("\nFast policy internals:")
        print(f"  Has actor_W: {hasattr(policy, 'actor_W')}")
        print(f"  Has actor_bias: {hasattr(policy, 'actor_bias')}")
        if hasattr(policy, "actor_W"):
            print(f"    actor_W shape: {policy.actor_W.shape}")
            print(f"    actor_W params: {policy.actor_W.numel()}")
        if hasattr(policy, "actor_bias"):
            print(f"    actor_bias shape: {policy.actor_bias.shape}")
        if hasattr(policy, "action_embeddings"):
            print(f"  Action embeddings shape: {policy.action_embeddings.weight.shape}")
            print(f"  Action embeddings params: {policy.action_embeddings.weight.numel()}")

    # Check for pufferlib Fast methods
    if agent.policy.__class__.__name__ == "Fast":
        print("\nFast wrapper details:")
        print(f"  Has activate_action_embeddings: {hasattr(agent.policy, 'activate_action_embeddings')}")
        print(f"  Has _convert_logit_index_to_action: {hasattr(agent.policy, '_convert_logit_index_to_action')}")

    # LSTM details
    if hasattr(agent.policy, "lstm"):
        lstm = agent.policy.lstm
        print("\nLSTM details:")
        print(f"  Type: {type(lstm)}")
        print(f"  Layers: {lstm.num_layers if hasattr(lstm, 'num_layers') else 'N/A'}")
        print(f"  Hidden size: {lstm.hidden_size if hasattr(lstm, 'hidden_size') else 'N/A'}")
        lstm_params = sum(p.numel() for p in lstm.parameters())
        print(f"  Total LSTM params: {lstm_params:,}")
    elif hasattr(agent.policy, "components") and "_core_" in agent.policy.components:
        core = agent.policy.components["_core_"]
        print("\nCore (LSTM) component:")
        print(f"  Type: {type(core)}")
        if hasattr(core, "_net"):
            lstm = core._net
            print(f"  LSTM type: {type(lstm)}")
            print(f"  Layers: {lstm.num_layers if hasattr(lstm, 'num_layers') else 'N/A'}")
            print(f"  Hidden size: {lstm.hidden_size if hasattr(lstm, 'hidden_size') else 'N/A'}")
            lstm_params = sum(p.numel() for p in lstm.parameters())
            print(f"  Total LSTM params: {lstm_params:,}")


def main():
    print("Creating mock environment...")
    env = create_mock_env()

    system_cfg = SystemConfig(device="cpu")

    # Create YAML agent
    print("\n" + "=" * 60)
    print("Creating YAML agent (agent=fast)")
    print("=" * 60)
    yaml_cfg = OmegaConf.load("configs/agent/fast.yaml")
    yaml_agent = MettaAgent(env, system_cfg, yaml_cfg)

    # Initialize to environment
    print("\nInitializing YAML agent to environment...")
    features = env.get_observation_features()
    yaml_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))

    # Create py_agent
    print("\n" + "=" * 60)
    print("Creating py_agent (py_agent=fast)")
    print("=" * 60)
    py_agent_cfg = OmegaConf.create(
        {
            "agent_type": "fast",
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "observations": {"obs_key": "grid_obs"},
        }
    )
    py_agent = MettaAgent(env, system_cfg, py_agent_cfg)

    # Initialize to environment
    print("\nInitializing py_agent to environment...")
    py_agent.initialize_to_environment(features, env.action_names, env.max_action_args, torch.device("cpu"))

    # Analyze both
    analyze_agent(yaml_agent, "YAML agent=fast")
    analyze_agent(py_agent, "py_agent=fast")

    # Direct comparison
    print(f"\n{'=' * 60}")
    print("DIRECT COMPARISON")
    print(f"{'=' * 60}")
    yaml_params = sum(p.numel() for p in yaml_agent.parameters())
    py_params = sum(p.numel() for p in py_agent.parameters())
    print(f"YAML agent params: {yaml_params:,}")
    print(f"py_agent params: {py_params:,}")
    print(f"Difference: {abs(yaml_params - py_params):,}")

    if yaml_params != py_params:
        print("\nWARNING: Parameter counts don't match!")
        print("This suggests the architectures are different.")


if __name__ == "__main__":
    main()
