#!/usr/bin/env -S uv run
"""
Launch a matrix of skypilot test jobs for validation.

This script launches 9 test jobs:
- 3 node configurations: 1, 2, 4 nodes
- 3 test conditions: normal completion, heartbeat timeout, runtime timeout
- Each with CI tests enabled for one job per node configuration
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gitta as git
from devops.skypilot.utils.job_helpers import (
    check_git_state,
    get_job_id_from_request_id,
    get_request_id_from_launch_output,
)
from metta.common.util.text_styles import bold, cyan, green, red, yellow

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
    nodes: int,
    condition_config: dict,
    run_name: str,
    enable_ci_tests: bool = False,
    skip_git_check: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
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

    if skip_git_check:
        cmd.append("--skip-git-check")

    # Display launch info
    print(f"\n{bold('Launching job:')} {yellow(run_name)}")
    print(f"  {cyan('Nodes:')} {nodes}")
    print(f"  {cyan('Condition:')} {condition_config['name']}")
    print(f"  {cyan('CI Tests:')} {'Yes' if enable_ci_tests else 'No'}")

    try:
        # Launch the job and capture output - removed check=True
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Combine stdout and stderr for parsing
        full_output = result.stdout + "\n" + result.stderr

        # Extract request ID from output using helper
        request_id = get_request_id_from_launch_output(full_output)

        if request_id:
            print(f"  {green('✅ Launched successfully')} - Request ID: {yellow(request_id)}")

            # Try to get job ID using helper
            job_id = get_job_id_from_request_id(request_id)

            if job_id:
                print(f"  {green('✅ Job ID retrieved:')} {yellow(job_id)}")
                return job_id, request_id
            else:
                print(f"  {cyan('⚠️  Job ID not available yet (may need more time)')}")
                return None, request_id
        else:
            print(f"  {red('❌ Failed to launch job')}")
            if result.stderr:
                print(f"  {red('Error:')} {result.stderr}")
            return None, None

    except Exception as e:
        print(f"  {red('❌ Failed to launch job')}")
        print(f"  {red('Error:')} {str(e)}")
        return None, None


def print_summary_table(launched_jobs: list, failed_launches: list) -> None:
    """Print a nice summary table of all jobs."""
    if not launched_jobs:
        return

    print("\n" + bold("Launched Jobs Summary:"))
    print("─" * 80)
    print(f"{'Nodes':^6} │ {'Condition':^20} │ {'CI':^4} │ {'Job ID':^10} │ {'Request ID'}")
    print("─" * 80)

    for job in launched_jobs:
        nodes = job["nodes"]
        condition = job["condition_name"][:20]
        ci = "✓" if job["ci_tests_enabled"] else ""
        job_id = job["job_id"] or "pending"
        request_id = job["request_id"][:8] + "..."

        print(f"{nodes:^6} │ {condition:^20} │ {ci:^4} │ {job_id:^10} │ {request_id}")

    print("─" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Launch skypilot test matrix")
    parser.add_argument("--base-name", default="skypilot_test", help="Base name for the test runs")
    parser.add_argument("--output-file", default="skypilot_test_jobs.json", help="Output file for job information")
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")

    args = parser.parse_args()

    # Check git state before launching any jobs
    if not args.skip_git_check:
        print(f"\n{bold('Checking git state...')}")
        commit_hash = git.get_current_commit()
        error_message = check_git_state(commit_hash)
        if error_message:
            print(error_message)
            print("  - Skip check: add --skip-git-check flag")
            sys.exit(1)
        print(f"{green('✅ Git state is clean')}")

    # Show test matrix
    print(f"\n{bold('=== Skypilot Test Matrix ===')}")
    print(f"{cyan('Node configurations:')} {NODE_CONFIGS}")
    print(f"{cyan('Test conditions:')}")
    for _key, config in TEST_CONDITIONS.items():
        print(f"  • {yellow(config['name'])}: {config['description']}")
    print(f"\n{cyan('Total jobs to launch:')} {len(NODE_CONFIGS) * len(TEST_CONDITIONS)}")
    print(f"{cyan('Output file:')} {args.output_file}")

    # Track all launched jobs
    launched_jobs = []
    failed_launches = []

    # Launch jobs
    for nodes in NODE_CONFIGS:
        for condition_key, condition_config in TEST_CONDITIONS.items():
            # Enable CI tests for the first condition of each node configuration
            enable_ci_tests = condition_key == "runtime_timeout"

            run_name = generate_run_name(args.base_name, nodes, condition_key, ci_test=enable_ci_tests)

            job_id, request_id = launch_job(
                nodes=nodes,
                condition_config=condition_config,
                run_name=run_name,
                enable_ci_tests=enable_ci_tests,
                skip_git_check=args.skip_git_check,
            )

            job_info = {
                "job_id": job_id,
                "request_id": request_id,
                "run_name": run_name,
                "nodes": nodes,
                "condition": condition_key,
                "condition_name": condition_config["name"],
                "condition_description": condition_config["description"],
                "ci_tests_enabled": enable_ci_tests,
                "launch_time": datetime.now().isoformat(),
                "success": request_id is not None,
            }

            if request_id:
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
        },
        "launched_jobs": launched_jobs,
        "failed_launches": failed_launches,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.basename(args.output_file)
    output_path = Path(script_dir) / output_filename
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{bold('=== Launch Summary ===')}")
    if len(launched_jobs) > 0:
        print(f"{green('Successfully launched:')} {len(launched_jobs)} jobs")
    if len(failed_launches) > 0:
        print(f"{red('Failed to launch:')} {len(failed_launches)} jobs")
    print(f"{cyan('Results saved to:')} {output_path.absolute()}")

    # Print nice summary table
    if launched_jobs:
        print_summary_table(launched_jobs, failed_launches)

    if failed_launches:
        print(f"\n{red('Failed launches:')}")
        for job in failed_launches:
            ci_marker = " (CI)" if job["ci_tests_enabled"] else ""
            print(f"  • {job['run_name']} ({job['nodes']} nodes, {job['condition_name']}{ci_marker})")
        sys.exit(1)


if __name__ == "__main__":
    main()
