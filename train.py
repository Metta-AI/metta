#!/usr/bin/env python3
"""
Metta training script - clean API without Hydra.

Usage:
    # Train with defaults
    python train.py

    # Train with custom settings
    python train.py --timesteps 1000000 --batch-size 512 --num-agents 4

    # Evaluate a checkpoint
    python train.py --mode eval --checkpoint path/to/checkpoint.pt
"""

import argparse

import metta_api as metta


def train(args):
    """Train a Metta agent."""
    # Setup environment
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        seed=args.seed,
        vectorization=args.vectorization,
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("train")

    # Import pufferlib for advantage computation
    try:
        from pufferlib import _C  # noqa: F401
    except ImportError:
        raise ImportError("Failed to import pufferlib C extensions. Try installing with --no-build-isolation")

    logger.info(f"Training run: {args.run}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Quick train wrapper
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
        env_width=args.env_width,
        env_height=args.env_height,
        logger=logger,
    )

    logger.info(f"Training complete! Final checkpoint: {checkpoint_path}")
    return checkpoint_path


def evaluate(args):
    """Evaluate a trained checkpoint."""
    # Setup environment
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        vectorization="multiprocessing",  # Better for eval
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("eval")

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    # Quick eval wrapper
    results = metta.quick_eval(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        device=config["device"],
        vectorization="multiprocessing",
        env_width=args.env_width,
        env_height=args.env_height,
        logger=logger,
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Average reward: {results['avg_reward']:.4f}")
    if "episode_lengths" in results:
        print(f"Average episode length: {results['avg_episode_length']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Metta - Clean RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode to run")

    # Common arguments
    parser.add_argument("--run", default="default_run", help="Experiment name")
    parser.add_argument("--device", help="Device (defaults to cuda if available)")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents per environment")
    parser.add_argument("--env-width", type=int, default=15, help="Environment width")
    parser.add_argument("--env-height", type=int, default=10, help="Environment height")

    # Training arguments
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps to train")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--vectorization", default="serial", choices=["serial", "multiprocessing", "ray"], help="Vectorization mode"
    )

    # Evaluation arguments
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate (for eval mode)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of environments for evaluation")

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == "train":
        checkpoint = train(args)
        print(f"\nTo evaluate: python train.py --mode eval --checkpoint {checkpoint}")
    elif args.mode == "eval":
        if not args.checkpoint:
            parser.error("--checkpoint is required for eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()
