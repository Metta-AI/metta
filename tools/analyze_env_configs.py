#!/usr/bin/env -S uv run
"""
Analyze environment configurations from wandb artifacts.

This script helps retrieve and analyze the environment configurations that were saved
as artifacts during training runs, especially useful for understanding how curricula
changed configs dynamically.
"""

import argparse
import json
import os
from typing import Dict

import wandb


def download_env_config_artifact(run_path: str, output_dir: str) -> str:
    """Download environment configuration artifact from a wandb run."""
    api = wandb.Api()
    run = api.run(run_path)

    # Find the env config artifact
    artifacts = run.logged_artifacts()
    env_config_artifacts = [a for a in artifacts if a.type == "environment_configs"]

    if not env_config_artifacts:
        raise ValueError(f"No environment configuration artifacts found for run {run_path}")

    # Get the latest version
    artifact = env_config_artifacts[-1]
    print(f"Downloading artifact: {artifact.name} (version {artifact.version})")

    # Download the artifact
    artifact_dir = artifact.download(root=output_dir)

    # Find the JSON file
    json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError("No JSON file found in artifact")

    return os.path.join(artifact_dir, json_files[0])


def analyze_env_configs(config_file: str) -> Dict:
    """Analyze environment configurations from a JSON file."""
    with open(config_file, "r") as f:
        data = json.load(f)

    history = data.get("history", [])
    task_counts = data.get("task_counts", {})
    curriculum_stats = data.get("curriculum_stats", {})

    # Extract unique configurations
    unique_configs = {}
    config_changes = []

    for entry in history:
        task_id = entry["task_id"]
        config = entry["config"]

        # Create a config signature
        config_str = json.dumps(config, sort_keys=True)

        if task_id not in unique_configs:
            unique_configs[task_id] = {}

        if config_str not in unique_configs[task_id]:
            unique_configs[task_id][config_str] = {
                "first_seen_epoch": entry["epoch"],
                "first_seen_step": entry["agent_step"],
                "count": 0,
                "config": config,
            }

            # Track when configs changed
            if len(config_changes) == 0 or config_changes[-1]["config_str"] != config_str:
                config_changes.append(
                    {"epoch": entry["epoch"], "step": entry["agent_step"], "task_id": task_id, "config_str": config_str}
                )

        unique_configs[task_id][config_str]["count"] += 1

    # Extract key parameters
    key_params = {}
    for entry in history:
        config = entry["config"]
        if "game" in config:
            game_cfg = config["game"]

            params = {
                "max_steps": game_cfg.get("max_steps"),
                "num_agents": game_cfg.get("num_agents"),
            }

            if "agent" in game_cfg:
                agent_cfg = game_cfg["agent"]
                params.update(
                    {
                        "freeze_duration": agent_cfg.get("freeze_duration"),
                        "default_resource_limit": agent_cfg.get("default_resource_limit"),
                    }
                )

                if "rewards" in agent_cfg:
                    for k, v in agent_cfg["rewards"].items():
                        params[f"reward_{k}"] = v

            key = f"{entry['task_id']}_epoch_{entry['epoch']}"
            key_params[key] = params

    return {
        "total_configs": len(history),
        "unique_tasks": len(task_counts),
        "task_counts": task_counts,
        "curriculum_stats": curriculum_stats,
        "config_changes": config_changes,
        "unique_configs_per_task": {task_id: len(configs) for task_id, configs in unique_configs.items()},
        "key_parameters": key_params,
    }


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    print("\n=== Environment Configuration Analysis ===")
    print(f"Total configurations logged: {analysis['total_configs']}")
    print(f"Unique tasks: {analysis['unique_tasks']}")

    print("\n=== Task Frequencies ===")
    for task_id, count in analysis["task_counts"].items():
        print(f"  {task_id}: {count} times")

    print("\n=== Configuration Changes ===")
    print(f"Total configuration changes: {len(analysis['config_changes'])}")
    for i, change in enumerate(analysis["config_changes"][:10]):  # Show first 10
        print(f"  Change {i + 1}: Epoch {change['epoch']}, Step {change['step']}, Task: {change['task_id']}")

    if len(analysis["config_changes"]) > 10:
        print(f"  ... and {len(analysis['config_changes']) - 10} more changes")

    print("\n=== Unique Configurations per Task ===")
    for task_id, count in analysis["unique_configs_per_task"].items():
        print(f"  {task_id}: {count} unique configurations")

    # Show a sample of key parameters
    print("\n=== Sample Key Parameters ===")
    sample_keys = list(analysis["key_parameters"].keys())[:5]
    for key in sample_keys:
        params = analysis["key_parameters"][key]
        print(f"\n  {key}:")
        for param, value in params.items():
            if value is not None:
                print(f"    {param}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Analyze environment configurations from wandb artifacts")
    parser.add_argument("run_path", help="W&B run path (e.g., entity/project/run_id)")
    parser.add_argument("--output-dir", default="./env_config_analysis", help="Output directory for artifacts")
    parser.add_argument("--export-csv", help="Export key parameters to CSV file")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Download artifact
        config_file = download_env_config_artifact(args.run_path, args.output_dir)
        print(f"\nConfiguration file downloaded to: {config_file}")

        # Analyze configurations
        analysis = analyze_env_configs(config_file)

        # Print analysis
        print_analysis(analysis)

        # Export to CSV if requested
        if args.export_csv:
            import pandas as pd

            # Convert key parameters to DataFrame
            params_list = []
            for key, params in analysis["key_parameters"].items():
                row = {"config_key": key}
                row.update(params)
                params_list.append(row)

            df = pd.DataFrame(params_list)
            df.to_csv(args.export_csv, index=False)
            print(f"\nKey parameters exported to: {args.export_csv}")

        # Save analysis results
        analysis_file = os.path.join(args.output_dir, "analysis_results.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis results saved to: {analysis_file}")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
