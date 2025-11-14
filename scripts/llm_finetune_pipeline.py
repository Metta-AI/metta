#!/usr/bin/env python3
"""Complete pipeline for LLM fine-tuning on MettaGrid with return conditioning."""

import argparse
import json
from pathlib import Path

import numpy as np

from metta.llm.observation_encoder import ObservationEncoder
from metta.llm.tinker_dataset_builder import TinkerDatasetBuilder
from metta.llm.trajectory_collector import TrajectoryCollector
from mettagrid.builder.envs import make_arena
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./llm_training_data", help="Output directory")
    parser.add_argument("--episodes-per-policy", type=int, default=200, help="Episodes per policy")
    parser.add_argument("--use-return-conditioning", action="store_true", default=True)

    # Policy URIs for diverse data collection
    parser.add_argument("--expert-policy", help="URI for expert policy (high reward)")
    parser.add_argument("--medium-policy", help="URI for medium policy")
    parser.add_argument("--weak-policy", help="URI for weak policy")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Setup environment
    print("Setting up MettaGrid environment...")
    arena_env = make_arena(num_agents=24)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(arena_env)
    encoder = ObservationEncoder(policy_env_info)

    # Step 2: Collect diverse trajectories
    print("\n=== Collecting Diverse Trajectories ===")
    collector = TrajectoryCollector(arena_env, encoder)

    # Collect from multiple policies
    policy_uris = {}
    if args.expert_policy:
        policy_uris["expert"] = args.expert_policy
    if args.medium_policy:
        policy_uris["medium"] = args.medium_policy
    if args.weak_policy:
        policy_uris["weak"] = args.weak_policy

    if not policy_uris:
        print("ERROR: No policy URIs provided. Use --expert-policy, --medium-policy, or --weak-policy")
        return

    all_episodes = collector.collect_diverse_dataset(
        policy_uris=policy_uris,
        episodes_per_policy=args.episodes_per_policy,
    )

    # Print dataset statistics
    returns = [ep.total_reward for ep in all_episodes]
    print("\n=== Dataset Statistics ===")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total steps: {sum(len(ep.steps) for ep in all_episodes)}")
    print("Return statistics:")
    print(f"  Min: {min(returns):.1f}")
    print(f"  Max: {max(returns):.1f}")
    print(f"  Mean: {np.mean(returns):.1f}")
    print(f"  Median: {np.median(returns):.1f}")
    print(f"  Std: {np.std(returns):.1f}")

    # Step 3: Build Tinker dataset
    print("\n=== Building Tinker Dataset ===")
    builder = TinkerDatasetBuilder()
    dataset = builder.build_dataset(
        episodes=all_episodes,
        use_return_conditioning=args.use_return_conditioning,
    )

    # Save dataset
    train_path = output_dir / "train.jsonl"
    builder.save_dataset(dataset, str(train_path))

    # Save metadata
    metadata = {
        "num_episodes": len(all_episodes),
        "num_steps": sum(len(ep.steps) for ep in all_episodes),
        "return_stats": {
            "min": float(min(returns)),
            "max": float(max(returns)),
            "mean": float(np.mean(returns)),
            "median": float(np.median(returns)),
            "std": float(np.std(returns)),
        },
        "policies": list(policy_uris.keys()),
        "use_return_conditioning": args.use_return_conditioning,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Complete! ===")
    print(f"Dataset saved to: {train_path}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")
    print("\nNext steps:")
    print("1. Set TINKER_API_KEY environment variable")
    print(f"2. Run: uv run python metta/llm/finetune_with_tinker.py --dataset {train_path}")


if __name__ == "__main__":
    main()
