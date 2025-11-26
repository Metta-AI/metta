#!/usr/bin/env python3
"""Download W&B metrics data for all training runs and cache locally."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Install with: uv add wandb", file=sys.stderr)
    sys.exit(1)


def get_run_metrics(run_id: str, project: str = "metta") -> dict[str, Any]:
    """Download all relevant metrics from a W&B run."""
    try:
        api = wandb.Api(timeout=120)
        run = api.run(f"{project}/{run_id}")
        history = run.history()

        # Extract per-env metrics
        per_env_metrics: dict[str, dict[str, Any]] = {}

        # Find all per-label reward columns
        reward_cols = [col for col in history.columns if col.startswith("env_per_label_rewards/")]
        for col in reward_cols:
            # Extract mission name from "env_per_label_rewards/mission_name" or "env_per_label_rewards/mission_name.avg"
            if ".avg" in col:
                mission = col.split("env_per_label_rewards/")[1].split(".avg")[0]
                metric_type = "avg_reward"
            else:
                mission = col.split("env_per_label_rewards/")[1]
                metric_type = "reward"

            if mission not in per_env_metrics:
                per_env_metrics[mission] = {}

            per_env_metrics[mission][metric_type] = float(history[col].mean())

        # Find per-label chest deposit metrics
        chest_cols = [col for col in history.columns if col.startswith("env_per_label_chest_deposits/")]
        for col in chest_cols:
            mission = col.split("env_per_label_chest_deposits/")[1]
            if mission not in per_env_metrics:
                per_env_metrics[mission] = {}
            # Use average number of chest deposits per episode (not just rate)
            per_env_metrics[mission]["chest_deposits_avg"] = float(history[col].mean())
            # Also keep the rate for reference
            per_env_metrics[mission]["chest_deposits_rate"] = float((history[col] > 0).mean())

        # Global metrics
        global_metrics = {
            "heart_gained_rate": 0.0,
            "chest_deposited_rate": 0.0,
            "avg_reward": 0.0,
        }

        # For heart gains, we need to check if ANY agent gained a heart in an episode
        # env_agent/heart.gained is per-agent, so we check if the sum across agents > 0
        if "env_agent/heart.gained" in history.columns:
            # Check if any agent gained a heart (value > 0 means at least one agent gained)
            # This gives us the rate of episodes where at least one agent gained a heart
            global_metrics["heart_gained_rate"] = float((history["env_agent/heart.gained"] > 0).mean())
        # Alternative: try to find game-level heart metrics if they exist
        # Note: There might not be a direct game-level heart.gained metric

        if "env_game/chest.heart.deposited" in history.columns:
            global_metrics["chest_deposited_rate"] = float((history["env_game/chest.heart.deposited"] > 0).mean())

        if "overview/reward" in history.columns:
            # Use the most recent (last) reward value instead of averaging
            reward_series = history["overview/reward"]
            # Drop NaN values and get the last non-NaN value
            reward_series_clean = reward_series.dropna()
            if len(reward_series_clean) > 0:
                global_metrics["avg_reward"] = float(reward_series_clean.iloc[-1])
            else:
                global_metrics["avg_reward"] = 0.0
        else:
            # Sum average rewards across all missions, use last value
            reward_avg_cols = [col for col in reward_cols if ".avg" in col]
            if reward_avg_cols:
                reward_sums = history[reward_avg_cols].sum(axis=1)
                reward_sums_clean = reward_sums.dropna()
                if len(reward_sums_clean) > 0:
                    global_metrics["avg_reward"] = float(reward_sums_clean.iloc[-1])
                else:
                    global_metrics["avg_reward"] = 0.0

        return {
            "run_id": run_id,
            "global_metrics": global_metrics,
            "per_env_metrics": per_env_metrics,
        }
    except Exception as e:
        print(f"Error fetching data for {run_id}: {e}", file=sys.stderr)
        return {
            "run_id": run_id,
            "global_metrics": {"heart_gained_rate": 0.0, "chest_deposited_rate": 0.0, "avg_reward": 0.0},
            "per_env_metrics": {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Download W&B metrics data for training runs")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="List of W&B run IDs to download (if not provided, reads from analysis JSON files)",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("run_analysis/json"),
        help="Directory containing analysis JSON files (to extract run IDs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("run_analysis/wandb_data"),
        help="Output directory for cached W&B data",
    )
    parser.add_argument("--project", type=str, default="metta", help="W&B project name")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of run IDs
    if args.run_ids:
        run_ids = args.run_ids
    else:
        # Extract run IDs from analysis JSON files
        run_ids = []
        for json_file in sorted(args.json_dir.glob("analysis_*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    run_id = data.get("run_id")
                    if run_id:
                        run_ids.append(run_id)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)

    if not run_ids:
        print("No run IDs found!", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading W&B data for {len(run_ids)} runs...")
    print("This may take a few minutes...\n")

    all_data = {}
    for i, run_id in enumerate(run_ids, 1):
        print(f"[{i}/{len(run_ids)}] Downloading {run_id}...", end=" ", flush=True)
        data = get_run_metrics(run_id, args.project)
        all_data[run_id] = data

        # Save individual file
        output_file = args.output_dir / f"{run_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        if "error" in data:
            print(f"ERROR: {data['error']}")
        else:
            num_envs = len(data.get("per_env_metrics", {}))
            print(f"✓ ({num_envs} environments)")

    # Save combined file
    combined_file = args.output_dir / "all_runs.json"
    with open(combined_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n✓ Downloaded data for {len(run_ids)} runs")
    print(f"✓ Individual files saved to {args.output_dir}/")
    print(f"✓ Combined file saved to {combined_file}")


if __name__ == "__main__":
    main()
