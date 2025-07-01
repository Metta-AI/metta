#!/usr/bin/env python3
"""Debug script to examine wandb runs and see what data is available."""

import wandb
import json
from pprint import pprint

def debug_wandb_runs():
    """Debug wandb runs to see what data is available."""

    # Initialize wandb API
    api = wandb.Api()

    # Get the most recent runs
    runs = api.runs("metta-research/metta", order="-created_at")
    runs = list(runs)[:5]  # Just look at first 5 runs

    print(f"Examining {len(runs)} most recent runs...")
    print("=" * 80)

    for i, run in enumerate(runs):
        print(f"\nRUN {i+1}: {run.name}")
        print("-" * 40)

        # Check if run has config
        config = run.config
        print(f"Has config: {config is not None}")
        if config:
            print("Config keys:", list(config.keys()))

            # Look for trainer config
            if "trainer" in config:
                print("Trainer config found!")
                trainer = config["trainer"]
                print("Trainer keys:", list(trainer.keys()))

                # Look for env_overrides
                if "env_overrides" in trainer:
                    print("env_overrides found!")
                    env_overrides = trainer["env_overrides"]
                    print("env_overrides keys:", list(env_overrides.keys()))
                else:
                    print("No env_overrides in trainer")
            else:
                print("No trainer in config")

            # Look for agent config
            if "agent" in config:
                print("Agent config found!")
                agent = config["agent"]
                print("Agent keys:", list(agent.keys()))
            else:
                print("No agent in config")
        else:
            print("No config found")

        # Check if run has history
        try:
            history = run.history()
            print(f"Has history: {not history.empty}")
            if not history.empty:
                print(f"History shape: {history.shape}")
                print("History columns:", list(history.columns))

                # Look for reward/hearts columns
                reward_cols = [col for col in history.columns if "reward" in col.lower() or "heart" in col.lower()]
                print(f"Reward/hearts columns: {reward_cols}")

                if reward_cols:
                    print(f"Sample reward data: {history[reward_cols[0]].head()}")
            else:
                print("History is empty")
        except Exception as e:
            print(f"Error getting history: {e}")

        # Check run summary
        summary = run.summary
        print(f"Has summary: {summary is not None}")
        if summary:
            print("Summary keys:", list(summary.keys()))

            # Look for reward in summary
            reward_keys = [k for k in summary.keys() if "reward" in k.lower()]
            if reward_keys:
                print(f"Reward keys in summary: {reward_keys}")
                for key in reward_keys:
                    print(f"  {key}: {summary[key]}")

        print()

if __name__ == "__main__":
    debug_wandb_runs()
