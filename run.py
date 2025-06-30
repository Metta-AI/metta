#!/usr/bin/env python
"""Train and evaluate Metta agents."""

import argparse

from metta import api as metta

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="default_run", help="Experiment name")
parser.add_argument("--device", default=None, help="Device (cuda/cpu/auto)")
args = parser.parse_args()

# Setup runtime config
config = metta.build_runtime_config(
    run=args.run,
    device=args.device,
    seed=0,
    vectorization="serial",
)
metta.setup_metta_environment(config)

logger = metta.get_logger("metta")

# Train
logger.info("\n🚀 Starting training phase...")
checkpoint_path = metta.quick_train(
    run_name=args.run,
    timesteps=1_000_000,
    batch_size=16_384,
    num_agents=2,
    num_workers=1,
    learning_rate=0.0004573146765703167,
    checkpoint_interval=30,
    device=config["device"],
    vectorization="serial",
    minibatch_size=512,
    bptt_horizon=16,
    target_kl=0.02,
    anneal_lr=True,
    lr_schedule_type="linear",
    warmup_steps=None,
    l2_init_weight_update_interval=0,
    grad_stats_interval=10,
    evaluate_interval=300,
    wandb_enabled=True,
    wandb_project="metta",
    wandb_entity=None,
    resume_from=None,
    logger=logger,
)

logger.info(f"\n✅ Training complete! Checkpoint saved: {checkpoint_path}")

# Evaluate
logger.info("\n🔍 Starting evaluation phase...")

eval_config = metta.build_runtime_config(
    run=args.run,
    device=config["device"],
    vectorization="multiprocessing",
)

results = metta.quick_sim(
    run_name=args.run,
    policy_uri=f"file://{checkpoint_path}",
    num_envs=32,
    num_episodes=10,
    num_agents=2,
    device=eval_config["device"],
    logger=logger,
)
