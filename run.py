#!/usr/bin/env python
"""Direct Metta execution script for training and simulation.

This script provides a simple command-line interface for:
- Training agents
- Running simulations
- Visualizing results

Examples:
    # Train with default settings
    python run.py

    # Train with custom parameters
    python run.py train --timesteps 100000

    # Run simulation
    python run.py sim --run myrun --policy-uri file://path/to/policy.pt
"""

import argparse
import json
import os

from metta import api as metta

# Parse arguments
parser = argparse.ArgumentParser(
    description="Metta - Clean RL training without Hydra",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

subparsers = parser.add_subparsers(dest="command", help="Command to run")

# Train command - only essential arguments
train_parser = subparsers.add_parser("train", help="Train a policy")
train_parser.add_argument("--run", default="default_run", help="Experiment name")
train_parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
train_parser.add_argument("--batch-size", type=int, default=16_384, help="Batch size")
train_parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
train_parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
train_parser.add_argument("--resume-from", default=None, help="Resume from checkpoint path")

# Sim command - minimal arguments
sim_parser = subparsers.add_parser("sim", help="Run simulation/evaluation")
sim_parser.add_argument("--run", required=True, help="Run name")
sim_parser.add_argument("--policy-uri", required=True, help="Policy URI (e.g., file://path/to/policy.pt)")

# Suite command - simplified
suite_parser = subparsers.add_parser("suite", help="Run full simulation suite evaluation")
suite_parser.add_argument("--policy", required=True, help="Path to policy checkpoint")

args = parser.parse_args()

# Default to train if no command
if not args.command:
    args.command = "train"
    # Set minimal defaults
    args.run = "default_run"
    args.timesteps = 1_000_000
    args.batch_size = 16_384
    args.device = None
    args.wandb = False
    args.resume_from = None

# Execute based on command
if args.command == "train":
    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        seed=0,
        vectorization="serial",
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("train")

    logger.info(f"Training run: {args.run}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Train with sensible defaults hardcoded
    checkpoint_path = metta.quick_train(
        run_name=args.run,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        num_agents=2,
        num_workers=1,
        learning_rate=0.0004573146765703167,
        checkpoint_interval=30,  # epochs
        device=config["device"],
        vectorization="serial",
        minibatch_size=512,
        bptt_horizon=16,
        # Enhanced features with defaults
        target_kl=0.02,  # Reasonable default for early stopping
        anneal_lr=True,  # Generally good practice
        lr_schedule_type="linear",
        warmup_steps=None,
        l2_init_weight_update_interval=0,
        grad_stats_interval=10,  # Log gradient stats every 10 epochs
        evaluate_interval=300,  # Evaluate every 300 epochs
        wandb_enabled=args.wandb,
        wandb_project="metta",
        wandb_entity=None,
        resume_from=args.resume_from,
        logger=logger,
    )

    logger.info(f"Training complete! Final checkpoint: {checkpoint_path}")
    print(f"\nTo evaluate: python run.py sim --run {args.run} --policy-uri file://{checkpoint_path}")

elif args.command == "sim":
    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=None,  # Auto-detect
        vectorization="multiprocessing",  # Better for eval
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("sim")

    logger.info(f"Evaluating policy: {args.policy_uri}")

    # If the policy URI is a relative path, make it absolute
    if args.policy_uri.startswith("file://./"):
        args.policy_uri = f"file://{os.path.abspath(args.policy_uri[7:])}"

    # Evaluate with reasonable defaults
    results = metta.quick_sim(
        run_name=args.run,
        policy_uri=args.policy_uri,
        num_envs=32,
        num_episodes=10,
        num_agents=2,
        device=config["device"],
        logger=logger,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

elif args.command == "suite":
    # Setup
    config = metta.build_runtime_config(
        run="simulation_suite",
        device=None,  # Auto-detect
        vectorization="multiprocessing",
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("suite")

    logger.info("Running full simulation suite evaluation")

    # Evaluate using the simulation suite
    results = metta.run_simulation_suite(
        policy_path=args.policy,
        suite_name="eval",
        num_episodes=10,
        num_envs=32,
        device=config["device"],
        logger=logger,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

else:
    parser.error(f"Unknown command: {args.command}")
