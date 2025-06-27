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
    python run.py train --timesteps 100000 --batch-size 1024

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

# Train command
train_parser = subparsers.add_parser("train", help="Train a policy")
train_parser.add_argument("--run", default="default_run", help="Experiment name")
train_parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps (default: 1M for testing)")
train_parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
train_parser.add_argument("--learning-rate", type=float, default=0.0004573146765703167, help="Learning rate")
train_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
train_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
train_parser.add_argument("--checkpoint-interval", type=int, default=30, help="Checkpoint interval in epochs")
train_parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
train_parser.add_argument("--vectorization", default="serial", help="Vectorization mode")
train_parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
train_parser.add_argument("--bptt-horizon", type=int, default=16, help="BPTT horizon")
# Enhanced features
train_parser.add_argument("--target-kl", type=float, default=None, help="Target KL for early stopping")
train_parser.add_argument("--anneal-lr", action="store_true", help="Enable learning rate annealing")
train_parser.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine"], help="LR schedule type")
train_parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps for LR scheduler")
train_parser.add_argument("--l2-weight-interval", type=int, default=0, help="L2 weight update interval")
train_parser.add_argument("--grad-stats-interval", type=int, default=0, help="Gradient stats logging interval")
train_parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
train_parser.add_argument("--wandb-project", default="metta", help="Wandb project name")
train_parser.add_argument("--wandb-entity", default=None, help="Wandb entity")
train_parser.add_argument("--resume-from", default=None, help="Resume from checkpoint path")
train_parser.add_argument("--evaluate-interval", type=int, default=0, help="Evaluation interval in epochs")

# Sim command
sim_parser = subparsers.add_parser("sim", help="Run simulation/evaluation")
sim_parser.add_argument("--run", required=True, help="Run name")
sim_parser.add_argument("--policy-uri", required=True, help="Policy URI (e.g., file://path/to/policy.pt)")
sim_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
sim_parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
sim_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
sim_parser.add_argument("--device", help="Device (defaults to cuda if available)")

# Suite command - run full simulation suite
suite_parser = subparsers.add_parser("suite", help="Run full simulation suite evaluation")
suite_parser.add_argument("--policy", required=True, help="Path to policy checkpoint")
suite_parser.add_argument("--suite-name", default="eval", help="Suite name")
suite_parser.add_argument("--num-episodes", type=int, default=10, help="Episodes per task")
suite_parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
suite_parser.add_argument("--device", help="Device (defaults to cuda if available)")

args = parser.parse_args()

# Default to train if no command
if not args.command:
    args.command = "train"
    args.run = "default_run"
    args.timesteps = 1_000_000
    args.batch_size = 16_384  # More reasonable for testing while following pattern
    args.learning_rate = 0.0004573146765703167
    args.num_agents = 2
    args.num_workers = 1
    args.checkpoint_interval = 30
    args.device = None
    args.seed = 0
    args.vectorization = "serial"
    args.minibatch_size = 512  # Adjusted proportionally
    args.bptt_horizon = 16
    # Enhanced features defaults
    args.target_kl = None
    args.anneal_lr = False
    args.lr_schedule = "linear"
    args.warmup_steps = None
    args.l2_weight_interval = 0
    args.grad_stats_interval = 0
    args.evaluate_interval = 0
    args.wandb = False
    args.wandb_project = "metta"
    args.wandb_entity = None
    args.resume_from = None

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
        minibatch_size=args.minibatch_size,
        bptt_horizon=args.bptt_horizon,
        # Enhanced features
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        lr_schedule_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        l2_init_weight_update_interval=args.l2_weight_interval,
        grad_stats_interval=args.grad_stats_interval,
        evaluate_interval=args.evaluate_interval,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        resume_from=args.resume_from,
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
        num_agents=args.num_agents,
        device=config["device"],
        logger=logger,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

elif args.command == "suite":
    # Setup
    config = metta.build_runtime_config(
        run="simulation_suite",  # Fixed: use a default run name
        device=args.device,
        vectorization="multiprocessing",  # Better for eval
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("suite")

    logger.info("Running full simulation suite evaluation")

    # Evaluate using the simulation suite
    results = metta.run_simulation_suite(
        policy_path=args.policy,
        suite_name=args.suite_name,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        device=config["device"],
        logger=logger,
    )

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

else:
    parser.error(f"Unknown command: {args.command}")
