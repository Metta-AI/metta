#!/usr/bin/env python
"""Debug script to compare YAML agent=fast vs py_agent=fast"""

import logging

import torch
from omegaconf import OmegaConf

from metta.agent.metta_agent import MettaAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.system_config import SystemConfig

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_env():
    """Create a basic MettaGrid environment for testing."""
    env_cfg = OmegaConf.create(
        {
            "seed": 1,
            "num_envs": 1,
            "mettagrid": {
                "game": {
                    "map_builder": {"type": "bsp"},
                    "random_seed": 1,
                    "agents": {"count": 1},
                    "time_limit": 100,
                }
            },
        }
    )

    # Load actual config
    base_cfg = OmegaConf.load("configs/env/mettagrid/arena/basic_easy_shaped.yaml")
    env_cfg = OmegaConf.merge(base_cfg, env_cfg)

    return MettaGridEnv(env_cfg.mettagrid, 1, "cpu")


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

    # Check for key attributes
    if hasattr(agent.policy, "components"):
        print(f"ComponentPolicy components: {list(agent.policy.components.keys())}")

    # List all modules
    print("\nAll modules:")
    for name, module in agent.named_modules():
        if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")

    # Check specific components for py_agent
    if hasattr(agent.policy, "policy"):
        policy = agent.policy.policy
        print("\nPy_agent specific checks:")
        print(f"  Has actor_W: {hasattr(policy, 'actor_W')}")
        print(f"  Has actor_bias: {hasattr(policy, 'actor_bias')}")
        if hasattr(policy, "actor_W"):
            print(f"    actor_W shape: {policy.actor_W.shape}")
        if hasattr(policy, "actor_bias"):
            print(f"    actor_bias shape: {policy.actor_bias.shape}")
        print(
            f"  Action embeddings shape: {policy.action_embeddings.weight.shape if hasattr(policy, 'action_embeddings') else 'N/A'}"
        )

    # LSTM details
    if hasattr(agent.policy, "lstm"):
        lstm = agent.policy.lstm
        print("\nLSTM details:")
        print(f"  Layers: {lstm.num_layers if hasattr(lstm, 'num_layers') else 'N/A'}")
        print(f"  Hidden size: {lstm.hidden_size if hasattr(lstm, 'hidden_size') else 'N/A'}")
        # Check bias initialization
        for name, param in lstm.named_parameters():
            if "bias" in name:
                print(f"  {name} mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
                break


def main():
    print("Creating environment...")
    env = create_env()

    system_cfg = SystemConfig(device="cpu")

    # Create YAML agent
    print("\n" + "=" * 60)
    print("Creating YAML agent (agent=fast)")
    print("=" * 60)
    yaml_cfg = OmegaConf.load("configs/agent/fast.yaml")
    yaml_agent = MettaAgent(env, system_cfg, yaml_cfg)

    # Initialize to environment
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

    # Test forward pass
    print(f"\n{'=' * 60}")
    print("TESTING FORWARD PASS")
    print(f"{'=' * 60}")

    test_obs = {"env_obs": torch.randint(0, 255, (4, 200, 3), dtype=torch.uint8)}

    try:
        yaml_out = yaml_agent(test_obs)
        print("YAML forward pass successful")
        print(f"  Output keys: {list(yaml_out.keys())}")
        if "values" in yaml_out:
            print(f"  Values shape: {yaml_out['values'].shape}")
        if "actions" in yaml_out:
            print(f"  Actions shape: {yaml_out['actions'].shape}")
    except Exception as e:
        print(f"YAML forward pass failed: {e}")

    try:
        py_out = py_agent(test_obs)
        print("py_agent forward pass successful")
        print(f"  Output keys: {list(py_out.keys())}")
        if "values" in py_out:
            print(f"  Values shape: {py_out['values'].shape}")
        if "actions" in py_out:
            print(f"  Actions shape: {py_out['actions'].shape}")
    except Exception as e:
        print(f"py_agent forward pass failed: {e}")


if __name__ == "__main__":
    main()
