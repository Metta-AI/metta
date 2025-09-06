#!/usr/bin/env -S uv run
"""
Launch a matrix of skypilot test jobs for validation.

This script launches 9 test jobs:
- 3 node configurations: 1, 2, 4 nodes
- 3 test conditions: normal completion, heartbeat timeout, runtime timeout
- Each with CI tests enabled for one job per node configuration
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Test matrix configuration
NODE_CONFIGS = [1, 2, 4]
TEST_CONDITIONS = {
    "normal_completion": {
        "name": "Normal Completion",
        "extra_args": ["--overrides", "trainer.total_timesteps=50000"],
        "description": "Exit normally after training completes",
    },
    "heartbeat_timeout": {
        "name": "Heartbeat Timeout",
        "extra_args": ["-hb", "1"],
        "description": "Exit based on missing heartbeats (1 second timeout)",
    },
    "runtime_timeout": {
        "name": "Runtime Timeout",
        "extra_args": ["-t", "0.03"],
        "description": "Exit based on timeout (0.03 hours = 1.8 minutes)",
    },
}

# Base configuration
BASE_MODULE = "experiments.recipes.arena_basic_easy_shaped.train"
BASE_ARGS = ["--no-spot", "--gpus=4"]


def generate_run_name(base: str, nodes: int, condition: str, ci_test: bool = False) -> str:
    """Generate a descriptive run name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ci_suffix = "_ci" if ci_test else ""
    return f"{base}_{nodes}n_{condition}{ci_suffix}_{timestamp}"


def launch_job(
    nodes: int, condition_config: dict, run_name: str, enable_ci_tests: bool = False, dry_run: bool = False
) -> Optional[str]:
    """Launch a single test job and return the job ID."""
    # Build the command
    cmd = [
        "devops/skypilot/launch.py",
        *BASE_ARGS,
        "--nodes",
        str(nodes),
        BASE_MODULE,
        "--args",
        f"run={run_name}",
        *condition_config["extra_args"],
    ]

    if enable_ci_tests:
        cmd.append("--run-ci-tests")

    print(f"\nLaunching job: {run_name}")
    print(f"  Nodes: {nodes}")
    print(f"  Condition: {condition_config['name']}")
    print(f"  CI Tests: {'Yes' if enable_ci_tests else 'No'}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Skipping actual launch")
        return f"dry_run_{run_name}"

    try:
        # Launch the job and capture output
        _result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Extract job ID from output (looking for the run name which becomes the job ID)
        # The job ID is typically the run name in skypilot
        job_id = run_name

        print(f"  ✅ Launched successfully - Job ID: {job_id}")
        return job_id

    except subprocess.CalledProcessError as e:
        print("  ❌ Failed to launch job")
        print(f"  Error: {e.stderr}")
        return None


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch skypilot test matrix")
    parser.add_argument("--base-name", default="skypilot_test", help="Base name for the test runs")
    parser.add_argument("--output-file", default="skypilot_test_jobs.json", help="Output file for job information")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without actually launching jobs")

    args = parser.parse_args()

    # Show test matrix
    print("=== Skypilot Test Matrix ===")
    print(f"Node configurations: {NODE_CONFIGS}")
    print("Test conditions:")
    for _key, config in TEST_CONDITIONS.items():
        print(f"  - {config['name']}: {config['description']}")
    print(f"\nTotal jobs to launch: {len(NODE_CONFIGS) * len(TEST_CONDITIONS)}")
    print(f"Output file: {args.output_file}")

    # Track all launched jobs
    launched_jobs = []
    failed_launches = []

    # Launch jobs
    for nodes in NODE_CONFIGS:
        for condition_idx, (condition_key, condition_config) in enumerate(TEST_CONDITIONS.items()):
            # Enable CI tests for the first condition of each node configuration
            enable_ci_tests = condition_idx == 0

            run_name = generate_run_name(args.base_name, nodes, condition_key, ci_test=enable_ci_tests)

            job_id = launch_job(
                nodes=nodes,
                condition_config=condition_config,
                run_name=run_name,
                enable_ci_tests=enable_ci_tests,
                dry_run=args.dry_run,
            )

            job_info = {
                "job_id": job_id,
                "run_name": run_name,
                "nodes": nodes,
                "condition": condition_key,
                "condition_name": condition_config["name"],
                "condition_description": condition_config["description"],
                "ci_tests_enabled": enable_ci_tests,
                "launch_time": datetime.now().isoformat(),
                "success": job_id is not None,
            }

            if job_id:
                launched_jobs.append(job_info)
            else:
                failed_launches.append(job_info)

    # Save results
    output_data = {
        "test_run_info": {
            "base_name": args.base_name,
            "launch_time": datetime.now().isoformat(),
            "total_jobs": len(launched_jobs) + len(failed_launches),
            "successful_launches": len(launched_jobs),
            "failed_launches": len(failed_launches),
            "dry_run": args.dry_run,
        },
        "launched_jobs": launched_jobs,
        "failed_launches": failed_launches,
    }

    output_path = Path(args.output_file)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n=== Launch Summary ===")
    print(f"Successfully launched: {len(launched_jobs)} jobs")
    print(f"Failed to launch: {len(failed_launches)} jobs")
    print(f"Results saved to: {output_path.absolute()}")

    if launched_jobs:
        print("\nLaunched job IDs:")
        for job in launched_jobs:
            ci_marker = " (CI)" if job["ci_tests_enabled"] else ""
            print(f"  - {job['job_id']} ({job['nodes']} nodes, {job['condition_name']}{ci_marker})")

    if failed_launches:
        print("\nFailed launches:")
        for job in failed_launches:
            print(f"  - {job['run_name']} ({job['nodes']} nodes, {job['condition_name']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
