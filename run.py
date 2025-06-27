#!/usr/bin/env python3
"""
Simple entry point for Metta without Hydra.

Usage:
    # Train (default)
    python run.py

    # Train with options
    python run.py train --run my_experiment --timesteps 1000000

    # Evaluate
    python run.py sim --run my_experiment --policy-uri file://./checkpoints/policy.pt
"""

import argparse
import json
import os

import metta_api as metta

# Parse arguments
parser = argparse.ArgumentParser(
    description="Metta - Clean RL training without Hydra",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

subparsers = parser.add_subparsers(dest="command", help="Command to run")

# Train command
train_parser = subparsers.add_parser("train", help="Train a policy")
train_parser.add_argument("--run", default="default_run", help="Experiment name")
train_parser.add_argument("--timesteps", type=int, default=10_000, help="Total timesteps")
train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
train_parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
train_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
train_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
train_parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval")
train_parser.add_argument("--device", help="Device (defaults to cuda if available)")
train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
train_parser.add_argument("--vectorization", default="serial", help="Vectorization mode")

# Sim command
sim_parser = subparsers.add_parser("sim", help="Evaluate a policy")
sim_parser.add_argument("--run", required=True, help="Experiment name")
sim_parser.add_argument("--policy-uri", required=True, help="Policy URI")
sim_parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
sim_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
sim_parser.add_argument("--device", help="Device (defaults to cuda if available)")

args = parser.parse_args()

# Default to train if no command
if not args.command:
    args.command = "train"
    args.run = "default_run"
    args.timesteps = 10_000
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.num_agents = 2
    args.num_workers = 1
    args.checkpoint_interval = 100
    args.device = None
    args.seed = 0
    args.vectorization = "serial"

# Execute based on command
if args.command == "train":
    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        seed=args.seed,
        vectorization=args.vectorization,
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("train")

    logger.info(f"Training run: {args.run}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Train using the quick_train wrapper
    checkpoint_path = metta.quick_train(
        run_name=args.run,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        device=config["device"],
        vectorization=args.vectorization,
        logger=logger,
    )

    logger.info(f"Training complete! Final checkpoint: {checkpoint_path}")
    print(f"\nTo evaluate: python run.py sim --run {args.run} --policy-uri file://{checkpoint_path}")

elif args.command == "sim":
    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        vectorization="multiprocessing",  # Better for eval
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("sim")

    logger.info(f"Evaluating policy: {args.policy_uri}")

    # If the policy URI is a relative path, make it absolute
    if args.policy_uri.startswith("file://./"):
        args.policy_uri = f"file://{os.path.abspath(args.policy_uri[7:])}"

    # Evaluate using the simulation suite
    results = metta.quick_sim(
        run_name=args.run,
        policy_uri=args.policy_uri,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=config["device"],
        logger=logger,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

else:
    parser.error(f"Unknown command: {args.command}")
