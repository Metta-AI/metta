#!/usr/bin/env python3
"""
Run the trader NPC experiment.
Usage: python run_trader_experiment.py [train|eval|play]
"""

import sys

from experiments.recipes.arena_trader_experiment import evaluate, play, train


def main():
    """Run trader experiment based on command line argument."""

    # Default to training if no argument provided
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        print("🚀 Training with trader NPCs...")
        trainer = train()

        # Override for reasonable local training
        trainer.trainer.total_timesteps = 100_000  # 100k steps (5-10 min locally)
        trainer.trainer.checkpoint_interval = 20_000
        trainer.trainer.eval_interval = 10_000
        trainer.trainer.log_interval = 1_000

        print(f"Training for {trainer.trainer.total_timesteps:,} steps...")
        print("This should take about 5-10 minutes locally")
        print("Check WandB for live metrics: https://wandb.ai/metta-research/metta")

        trainer.run()

    elif mode == "eval":
        print("📊 Evaluating...")
        policy = sys.argv[2] if len(sys.argv) > 2 else None
        evaluator = evaluate(policy_uri=policy)
        evaluator.run()

    elif mode == "play":
        print("🎮 Interactive play...")
        player = play()
        player.run()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run_trader_experiment.py [train|eval|play]")


if __name__ == "__main__":
    main()
