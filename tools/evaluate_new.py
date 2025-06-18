#!/usr/bin/env python
"""Simplified evaluation script using functional API.

This demonstrates evaluating policies without complex YAML configs.
"""

import argparse
import json
from typing import Any, Dict

import torch

# Import Python-based configs
from configs.python.environments import (
    all_eval_suite,
    memory_eval_suite,
    navigation_eval_suite,
    objectuse_eval_suite,
)
from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.train import evaluate_policy


def evaluate_checkpoint(
    checkpoint_path: str,
    env_suite: str = "all",
    device: str = "cuda",
    num_episodes: int = 10,
) -> Dict[str, Any]:
    """Evaluate a checkpoint on an environment suite.

    Args:
        checkpoint_path: Path to checkpoint file
        env_suite: Which evaluation suite to use
        device: Device to run on
        num_episodes: Episodes per environment

    Returns:
        Evaluation results
    """
    # Load agent from checkpoint
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    agent = load_agent_from_checkpoint(checkpoint_path, device)

    # Get environment suite
    if env_suite == "navigation":
        env_configs = navigation_eval_suite()
    elif env_suite == "memory":
        env_configs = memory_eval_suite()
    elif env_suite == "objectuse":
        env_configs = objectuse_eval_suite()
    elif env_suite == "all":
        env_configs = all_eval_suite()
    else:
        raise ValueError(f"Unknown environment suite: {env_suite}")

    print(f"📊 Evaluating on {len(env_configs)} environments")

    # Convert to simulation config format
    simulations = {}
    for env_config in env_configs:
        simulations[env_config.name] = {
            "env": env_config.to_dict(),
            "num_episodes": num_episodes,
        }

    sim_suite_config = SimulationSuiteConfig(
        {
            "name": f"{env_suite}_eval",
            "simulations": simulations,
        }
    )

    # Run evaluation
    result = evaluate_policy(agent, sim_suite_config, device)

    return {
        "checkpoint": checkpoint_path,
        "env_suite": env_suite,
        "overall_score": result.overall_score,
        "category_scores": result.category_scores,
        "individual_scores": result.individual_scores,
        "metadata": result.metadata,
    }


def evaluate_policy_uri(
    policy_uri: str,
    env_suite: str = "all",
    device: str = "cuda",
    num_episodes: int = 10,
    selector: str = "latest",
) -> Dict[str, Any]:
    """Evaluate a policy from a URI.

    Args:
        policy_uri: Policy URI (e.g., "wandb://run/...")
        env_suite: Which evaluation suite to use
        device: Device to run on
        num_episodes: Episodes per environment
        selector: Policy selection strategy

    Returns:
        Evaluation results
    """
    # Create minimal config for policy store
    config = {
        "device": device,
        "wandb": {"entity": "your-entity", "project": "metta"},
    }

    # Load policy
    print(f"📦 Loading policy from {policy_uri}")
    policy_store = PolicyStore(config, None)
    policy_pr = policy_store.policy(policy_uri, selector)
    agent = policy_pr.policy()

    # Get environment suite
    if env_suite == "navigation":
        env_configs = navigation_eval_suite()
    elif env_suite == "memory":
        env_configs = memory_eval_suite()
    elif env_suite == "objectuse":
        env_configs = objectuse_eval_suite()
    elif env_suite == "all":
        env_configs = all_eval_suite()
    else:
        raise ValueError(f"Unknown environment suite: {env_suite}")

    print(f"📊 Evaluating on {len(env_configs)} environments")

    # Convert to simulation config format
    simulations = {}
    for env_config in env_configs:
        simulations[env_config.name] = {
            "env": env_config.to_dict(),
            "num_episodes": num_episodes,
        }

    sim_suite_config = SimulationSuiteConfig(
        {
            "name": f"{env_suite}_eval",
            "simulations": simulations,
        }
    )

    # Run evaluation
    result = evaluate_policy(agent, sim_suite_config, device)

    return {
        "policy_uri": policy_uri,
        "policy_name": policy_pr.name,
        "env_suite": env_suite,
        "overall_score": result.overall_score,
        "category_scores": result.category_scores,
        "individual_scores": result.individual_scores,
        "metadata": result.metadata,
    }


def load_agent_from_checkpoint(checkpoint_path: str, device: str) -> MettaAgent:
    """Load an agent from a checkpoint file."""
    from configs.python.agents import simple_cnn_agent

    # Create agent with default config
    # In practice, you'd save the agent config with the checkpoint
    agent_config = simple_cnn_agent()
    agent = MettaAgent(**agent_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.to(device)
    agent.eval()

    return agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Metta policy")
    parser.add_argument("input", help="Checkpoint path or policy URI to evaluate")
    parser.add_argument(
        "--env-suite",
        type=str,
        default="all",
        choices=["navigation", "memory", "objectuse", "all"],
        help="Environment suite to evaluate on",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per environment")
    parser.add_argument("--selector", type=str, default="latest", help="Policy selection strategy for URIs")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Determine if input is a checkpoint file or URI
    if args.input.startswith(("wandb://", "s3://", "file://")):
        # Policy URI
        results = evaluate_policy_uri(
            args.input,
            env_suite=args.env_suite,
            device=args.device,
            num_episodes=args.num_episodes,
            selector=args.selector,
        )
    else:
        # Checkpoint file
        results = evaluate_checkpoint(
            args.input,
            env_suite=args.env_suite,
            device=args.device,
            num_episodes=args.num_episodes,
        )

    # Print results
    print("\n🎯 Evaluation Results:")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print("\nCategory Scores:")
    for category, score in results["category_scores"].items():
        print(f"  {category}: {score:.3f}")

    if len(results["individual_scores"]) <= 20:
        print("\nIndividual Environment Scores:")
        for env_name, score in sorted(results["individual_scores"].items()):
            print(f"  {env_name}: {score:.3f}")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
