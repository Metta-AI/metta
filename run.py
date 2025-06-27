#!/usr/bin/env python
"""Direct Metta training and evaluation script.

This script trains an agent and then automatically evaluates it.

Examples:
    # Train with default settings and evaluate
    python run.py

    # Train with custom parameters
    python run.py --timesteps 100000 --batch-size 16384

    # Train with wandb logging
    python run.py --wandb
"""

import argparse

from metta import api as metta

# Parse arguments - minimal set
parser = argparse.ArgumentParser(
    description="Metta - Train and evaluate RL agents",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument("--run", default="default_run", help="Experiment name")
parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--batch-size", type=int, default=16_384, help="Batch size")
parser.add_argument("--device", default=None, help="Device (cuda/cpu/auto)")
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--resume-from", default=None, help="Resume from checkpoint path")

args = parser.parse_args()

# Setup runtime config
config = metta.build_runtime_config(
    run=args.run,
    device=args.device,
    seed=0,
    vectorization="serial",  # Serial for training, will use multiprocessing for eval
)
metta.setup_metta_environment(config)

# Get logger
logger = metta.get_logger("metta")

logger.info("=" * 80)
logger.info(f"Starting Metta training run: {args.run}")
logger.info(f"  Device: {config['device']}")
logger.info(f"  Timesteps: {args.timesteps}")
logger.info(f"  Batch size: {args.batch_size}")
logger.info(f"  Wandb: {'enabled' if args.wandb else 'disabled'}")
logger.info("=" * 80)

# Train with sensible defaults
logger.info("\n🚀 Starting training phase...")
checkpoint_path = metta.quick_train(
    run_name=args.run,
    timesteps=args.timesteps,
    batch_size=args.batch_size,
    num_agents=2,
    num_workers=1,
    learning_rate=0.0004573146765703167,
    checkpoint_interval=30,  # Save every 30 epochs
    device=config["device"],
    vectorization="serial",
    minibatch_size=512,
    bptt_horizon=16,
    # Enhanced features with good defaults
    target_kl=0.02,  # Early stopping threshold
    anneal_lr=True,  # Linear LR decay
    lr_schedule_type="linear",
    warmup_steps=None,
    l2_init_weight_update_interval=0,
    grad_stats_interval=10,  # Log gradient stats every 10 epochs
    evaluate_interval=300,  # Evaluate during training every 300 epochs
    wandb_enabled=args.wandb,
    wandb_project="metta",
    wandb_entity=None,
    resume_from=args.resume_from,
    logger=logger,
)

logger.info(f"\n✅ Training complete! Checkpoint saved: {checkpoint_path}")

# Automatically evaluate the trained policy
logger.info("\n🔍 Starting evaluation phase...")

# Update config for evaluation
eval_config = metta.build_runtime_config(
    run=args.run,
    device=config["device"],  # Use same device
    vectorization="multiprocessing",  # Better for evaluation
)

# Run evaluation
results = metta.quick_sim(
    run_name=args.run,
    policy_uri=f"file://{checkpoint_path}",
    num_envs=32,  # More envs for faster evaluation
    num_episodes=10,  # Evaluate on 10 episodes
    num_agents=2,
    device=eval_config["device"],
    logger=logger,
)

# Print results
logger.info("\n📊 Evaluation Results:")
logger.info("=" * 80)
if results.get("policies"):
    policy_results = results["policies"][0]
    metrics = policy_results.get("metrics", {})

    logger.info(f"Policy: {policy_results['name']}")
    logger.info(f"  Episodes evaluated: {metrics.get('num_episodes', 0)}")
    logger.info(f"  Average reward: {metrics.get('avg_reward', 0):.2f}")
    logger.info(f"  Std deviation: {metrics.get('std_reward', 0):.2f}")
    logger.info(f"  Min reward: {metrics.get('min_reward', 0):.2f}")
    logger.info(f"  Max reward: {metrics.get('max_reward', 0):.2f}")

    if "avg_episode_length" in metrics:
        logger.info(f"  Average episode length: {metrics.get('avg_episode_length', 0):.1f}")
else:
    logger.info("No evaluation results available")

logger.info("=" * 80)
logger.info("\n🎉 All done! Your trained agent is ready.")
logger.info(f"   Checkpoint: {checkpoint_path}")
logger.info(f"   To run more evaluations: python -m metta.api quick_sim --policy-uri file://{checkpoint_path}")
