#!/usr/bin/env -S uv run
"""
Analyze environment configurations from wandb runs.

This script helps retrieve and analyze environment configurations and task selection
from wandb runs. With the new approach, task configs are stored in wandb.config and
task selection is tracked as metrics.
"""

import argparse
import json
import os
from typing import Dict, List

import wandb


def get_curriculum_data_from_run(run_path: str) -> Dict:
    """Get curriculum task configs and selection history from a wandb run."""
    api = wandb.Api()
    run = api.run(run_path)

    # Get task configs from wandb.config
    curriculum_data = run.config.get("curriculum", {})
    task_configs = curriculum_data.get("task_configs", {})

    if not task_configs:
        raise ValueError(f"No curriculum task configs found in run {run_path}")

    print(f"Found {len(task_configs)} task configs in wandb.config")

    # Get task selection history from metrics
    history = run.scan_history(keys=["curriculum/current_task_id", "curriculum/current_task_name"])

    history_list = list(history)
    print(f"Found {len(history_list)} task selection records")

    # Post-hoc parameter extraction from task configs
    enriched_history = []
    for entry in history_list:
        task_id = entry.get("curriculum/current_task_id")
        if task_id and task_id in task_configs:
            config = task_configs[task_id]
            # Extract key parameters from config
            game = config.get("game", {})
            agent = game.get("agent", {})

            enriched_entry = entry.copy()
            enriched_entry["freeze_duration"] = agent.get("freeze_duration")
            enriched_entry["max_steps"] = game.get("max_steps")
            enriched_entry["num_agents"] = game.get("num_agents")

            # Extract rewards
            rewards = agent.get("rewards", {})
            enriched_entry["reward_heart"] = rewards.get("heart")

            enriched_history.append(enriched_entry)

    return {
        "curriculum_data": curriculum_data,
        "task_configs": task_configs,
        "task_selection_history": enriched_history,
        "run_info": {"id": run.id, "name": run.name, "created_at": run.created_at, "url": run.url},
    }


def analyze_env_configs(data: Dict) -> Dict:
    """Analyze environment configurations and task selection patterns."""
    task_configs = data["task_configs"]
    history = data["task_selection_history"]

    # Task frequency analysis
    task_counts = {}
    for entry in history:
        task_id = entry.get("curriculum/current_task_id")
        if task_id:
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

    # Parameter analysis
    param_analysis = {}

    # Analyze freeze_duration values
    freeze_durations = [entry.get("freeze_duration") for entry in history if entry.get("freeze_duration") is not None]
    if freeze_durations:
        param_analysis["freeze_duration"] = {
            "min": min(freeze_durations),
            "max": max(freeze_durations),
            "unique_values": list(set(freeze_durations)),
            "count": len(freeze_durations),
        }

    # Analyze from task configs
    config_params = {}
    for task_id, config in task_configs.items():
        # Extract key parameters
        params = {}
        if "game" in config:
            game = config["game"]
            params["num_agents"] = game.get("num_agents")
            params["max_steps"] = game.get("max_steps")

            if "agent" in game:
                agent = game["agent"]
                params["freeze_duration"] = agent.get("freeze_duration")
                params["default_resource_limit"] = agent.get("default_resource_limit")

                if "rewards" in agent:
                    for reward_name, value in agent["rewards"].items():
                        params[f"reward_{reward_name}"] = value

        config_params[task_id] = {k: v for k, v in params.items() if v is not None}

    return {
        "total_task_selections": len(history),
        "unique_tasks": len(task_counts),
        "task_frequencies": task_counts,
        "parameter_analysis": param_analysis,
        "task_config_parameters": config_params,
        "curriculum_type": data["curriculum_data"].get("curriculum_type", "unknown"),
    }


def filter_runs_by_params(entity: str, project: str, **param_filters) -> List[str]:
    """Filter runs by environment parameters (post-hoc analysis)."""
    api = wandb.Api()

    # Get all runs and filter post-hoc
    runs = api.runs(f"{entity}/{project}")
    matching_runs = []

    for run in runs:
        curriculum_data = run.config.get("curriculum", {})
        task_configs = curriculum_data.get("task_configs", {})

        if not task_configs:
            continue

        # Check if any task in this run matches the criteria
        matches_criteria = False

        for task_id, config in task_configs.items():
            game = config.get("game", {})
            agent = game.get("agent", {})

            # Extract parameters for filtering
            params = {
                "freeze_duration": agent.get("freeze_duration"),
                "max_steps": game.get("max_steps"),
                "num_agents": game.get("num_agents"),
            }

            # Check each filter criterion
            task_matches = True
            for param_name, param_filter in param_filters.items():
                param_value = params.get(param_name)

                if param_value is None:
                    task_matches = False
                    break

                if isinstance(param_filter, dict):
                    if "$gt" in param_filter and not (param_value > param_filter["$gt"]):
                        task_matches = False
                        break
                    elif "$lt" in param_filter and not (param_value < param_filter["$lt"]):
                        task_matches = False
                        break
                else:
                    if param_value != param_filter:
                        task_matches = False
                        break

            if task_matches:
                matches_criteria = True
                break

        if matches_criteria:
            matching_runs.append(run.path)

    return matching_runs


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    print("\n=== Environment Configuration Analysis ===")
    print(f"Curriculum type: {analysis['curriculum_type']}")
    print(f"Total task selections: {analysis['total_task_selections']}")
    print(f"Unique tasks: {analysis['unique_tasks']}")

    print("\n=== Task Frequencies ===")
    for task_id, count in analysis["task_frequencies"].items():
        print(f"  {task_id}: {count} times")

    print("\n=== Parameter Analysis ===")
    for param_name, param_data in analysis["parameter_analysis"].items():
        print(f"\n  {param_name}:")
        for key, value in param_data.items():
            print(f"    {key}: {value}")

    print("\n=== Sample Task Configurations ===")
    for task_id, params in list(analysis["task_config_parameters"].items())[:3]:
        print(f"\n  {task_id}:")
        for param, value in params.items():
            print(f"    {param}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Analyze environment configurations from wandb runs")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze single run
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single run")
    analyze_parser.add_argument("run_path", help="W&B run path (e.g., entity/project/run_id)")
    analyze_parser.add_argument("--output-dir", default="./analysis", help="Output directory")

    # Filter runs by parameters
    filter_parser = subparsers.add_parser("filter", help="Filter runs by parameters")
    filter_parser.add_argument("entity", help="W&B entity")
    filter_parser.add_argument("project", help="W&B project")
    filter_parser.add_argument("--freeze-duration-gt", type=int, help="Filter freeze_duration > value")
    filter_parser.add_argument("--max-steps-lt", type=int, help="Filter max_steps < value")
    filter_parser.add_argument("--num-agents", type=int, help="Filter by exact num_agents")

    args = parser.parse_args()

    if args.command == "analyze":
        try:
            # Get data from run
            data = get_curriculum_data_from_run(args.run_path)

            # Analyze
            analysis = analyze_env_configs(data)

            # Print results
            print_analysis(analysis)

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            analysis_file = os.path.join(args.output_dir, "analysis_results.json")
            with open(analysis_file, "w") as f:
                json.dump({"analysis": analysis, "run_info": data["run_info"]}, f, indent=2, default=str)
            print(f"\nAnalysis results saved to: {analysis_file}")

        except Exception as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "filter":
        try:
            # Build parameter filters
            filters = {}
            if args.freeze_duration_gt:
                filters["freeze_duration"] = {"$gt": args.freeze_duration_gt}
            if args.max_steps_lt:
                filters["max_steps"] = {"$lt": args.max_steps_lt}
            if args.num_agents:
                filters["num_agents"] = args.num_agents

            # Find matching runs
            matching_runs = filter_runs_by_params(args.entity, args.project, **filters)

            print(f"\nFound {len(matching_runs)} runs matching criteria:")
            for run_path in matching_runs:
                print(f"  {run_path}")

        except Exception as e:
            print(f"Error: {e}")
            return 1

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
