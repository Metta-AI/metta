#!/usr/bin/env -S uv run
"""Launch curriculum experiment to compare different algorithms and task trees.

This script launches training jobs for different combinations of:
- Curriculum algorithms: discrete_random, learning_progress, prioritize_regressed
- Task trees: arena, navigation
"""

import argparse
import subprocess
import time
from datetime import datetime

# Map from (task_tree, algorithm) to curriculum config name
# Using curriculum store names
CURRICULUM_PATHS = {
    ("arena", "discrete_random"): "arena_random",
    ("arena", "learning_progress"): "arena_learning_progress",
    ("arena", "prioritize_regressed"): "arena_prioritize_regressed",
    ("navigation", "discrete_random"): "navigation_bucketed",
    ("navigation", "learning_progress"): "navigation_learning_progress",
    ("navigation", "prioritize_regressed"): "navigation_prioritize_regressed",
}

# Default algorithms and tasks to run
DEFAULT_ALGORITHMS = ["discrete_random", "learning_progress", "prioritize_regressed"]
DEFAULT_TASKS = ["arena", "navigation"]


def get_user():
    import os

    return os.environ.get("USER", "unknown")


def launch_training(task: str, algorithm: str) -> bool:
    """Launch a single training run."""
    user = get_user()
    key = (task, algorithm)
    if key not in CURRICULUM_PATHS:
        raise ValueError(f"No curriculum path for {task} + {algorithm}, skipping")

    curriculum_path = CURRICULUM_PATHS[key]
    timestamp = datetime.now().strftime("%m-%d")
    run_name = f"{user}.curriculum_exp.{task}.{algorithm}.{timestamp}"

    # Build command based on task type
    cmd = [
        "./devops/skypilot/launch.py",
        "train",
        f"run={run_name}",
        f"trainer.curriculum={curriculum_path}",
        f"sim={task}",
        "--no-spot",
    ]
    print(f"\nLaunching: {task} + {algorithm}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("✓ Successfully launched")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to launch")
        return False


def main():
    parser = argparse.ArgumentParser(description="Launch curriculum experiment")
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, help="Algorithms to test (default: all)")
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, help="Task trees to test (default: all)")

    args = parser.parse_args()

    # Use defaults if not specified
    algorithms = args.algorithms or DEFAULT_ALGORITHMS
    tasks = args.tasks or DEFAULT_TASKS

    # Get username
    # Launch all combinations
    total = len(algorithms) * len(tasks)
    launched = 0
    successful = 0

    print(f"Launching {total} training runs...")

    for task in tasks:
        for algorithm in algorithms:
            launched += 1
            if launch_training(task, algorithm):
                successful += 1

            # Small delay between launches
            if launched < total:
                time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Experiment launch complete: {successful}/{total} successful")
    print("Monitor with: sky status")


if __name__ == "__main__":
    main()
