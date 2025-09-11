#!/usr/bin/env -S uv run
"""
Test script for validating cluster configurations and exit conditions.

Usage:
    cluster_test.py --launch    # Launch 3x3 matrix test jobs
    cluster_test.py --check     # Check test results
    cluster_test.py --check -l  # Check with detailed logs
"""

import argparse
import sys

from devops.skypilot.utils.testing_helpers import SkyPilotJobChecker, SkyPilotTestLauncher, TestCondition
from metta.common.util.text_styles import bold, cyan, yellow

# Test matrix configuration
NODE_CONFIGS = [1, 2, 4]

TEST_CONDITIONS = {
    "normal_completion": TestCondition(
        name="Normal Completion",
        extra_args=["--overrides", "trainer.total_timesteps=50000"],
        description="Exit normally after training completes",
        ci=False,
    ),
    "heartbeat_timeout": TestCondition(
        name="Heartbeat Timeout",
        extra_args=["-hb", "1"],
        description="Exit based on missing heartbeats (1 second timeout)",
        ci=False,
    ),
    "runtime_timeout": TestCondition(
        name="Runtime Timeout",
        extra_args=["-t", "0.03"],
        description="Exit based on timeout (0.03 hours = 1.8 minutes)",
        ci=True,
    ),
}

# Base configuration
BASE_MODULE = "experiments.recipes.arena_basic_easy_shaped.train"
BASE_ARGS = ["--no-spot", "--gpus=4"]

# Default file name
DEFAULT_OUTPUT_FILE = "cluster_test_jobs.json"


def launch_tests(args):
    """Launch cluster test jobs."""
    # Create launcher
    launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

    # Check git state
    if not launcher.check_git_state():
        sys.exit(1)

    # Show test matrix
    print(f"\n{bold('=== Skypilot Cluster Test Matrix ===')}")
    print(f"{cyan('Node configurations:')} {NODE_CONFIGS}")
    print(f"{cyan('Test conditions:')}")
    for _key, condition in TEST_CONDITIONS.items():
        print(f"  • {yellow(condition.name)}: {condition.description}")
    print(f"\n{cyan('Total jobs to launch:')} {len(NODE_CONFIGS) * len(TEST_CONDITIONS)}")
    print(f"{cyan('Output file:')} {args.output_file}")

    # Launch jobs
    for nodes in NODE_CONFIGS:
        for condition_key, condition in TEST_CONDITIONS.items():
            enable_ci_tests = condition.ci

            # Generate unique run name
            suffix = f"{nodes}n_{condition_key}"
            if enable_ci_tests:
                suffix += "_ci"
            run_name = launcher.generate_run_name(suffix)

            # Test config for tracking
            test_config = {
                "nodes": nodes,
                "condition": condition_key,
                "condition_name": condition.name,
                "condition_description": condition.description,
                "ci_tests_enabled": enable_ci_tests,
            }

            # Launch the job
            launcher.launch_job(
                module=BASE_MODULE,
                run_name=run_name,
                base_args=BASE_ARGS + ["--nodes", str(nodes)],
                extra_args=condition.extra_args,
                test_config=test_config,
                enable_ci_tests=enable_ci_tests,
            )

    # Save results
    output_path = launcher.save_results(args.output_file)
    print(f"{cyan('Results saved to:')} {output_path.absolute()}")

    # Print summary
    launcher.print_summary()

    # Exit with error if any launches failed
    if launcher.failed_launches:
        sys.exit(1)


def check_tests(args):
    """Check cluster test results."""
    # Create checker
    checker = SkyPilotJobChecker(input_file=args.input_file)

    # Load jobs
    if not checker.load_jobs():
        print(f"Run '{sys.argv[0]} --launch' first to create the job file")
        sys.exit(1)

    # Get job count
    launched_jobs = checker.jobs_data.get("launched_jobs", [])
    if not launched_jobs:
        sys.exit(0)

    # Summary header
    test_info = checker.jobs_data.get("test_run_info", {})
    print(bold(f"\n=== Checking {len(launched_jobs)} Cluster Test Jobs ==="))
    print(f"{cyan('Test run:')} {test_info.get('base_name', 'Unknown')}")
    print(f"{cyan('Launch time:')} {test_info.get('launch_time', 'Unknown')}")
    print(f"{cyan('Input file:')} {args.input_file}")

    # Check job statuses
    checker.check_statuses()

    # Quick status summary first
    checker.print_quick_summary()

    # Parse job summaries from logs
    checker.parse_all_summaries(args.tail_lines)

    # Print detailed table
    checker.print_detailed_table()

    # Show detailed logs if requested
    if args.logs:
        checker.show_detailed_logs(args.tail_lines)

    # Print hints
    print(f"\n{bold('Hints:')}")
    print(f"  • Use {cyan('--check -l')} to view detailed job logs")
    print(f"  • Use {cyan('--check -n <lines>')} to change log lines to tail")
    print(f"  • Use {cyan('sky jobs logs <job_id>')} to view a single job's full log")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster configuration and exit condition test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script launches a 3x3 test matrix:
  - 3 node configurations: 1, 2, 4 nodes
  - 3 test conditions: normal completion, heartbeat timeout, runtime timeout
  - CI tests are enabled for runtime timeout jobs

Examples:
  %(prog)s --launch                    # Launch 9 test jobs
  %(prog)s --check                     # Check job results
  %(prog)s --check -l                  # Check with detailed logs
  %(prog)s --launch --base-name mytest # Launch with custom base name
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--launch", action="store_true", help="Launch test jobs")
    mode_group.add_argument("--check", action="store_true", help="Check test results")

    # Launch options
    launch_group = parser.add_argument_group("launch options")
    launch_group.add_argument("--base-name", default="cluster_test", help="Base name for test runs")
    launch_group.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Output JSON file")
    launch_group.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")

    # Check options
    check_group = parser.add_argument_group("check options")
    check_group.add_argument("-f", "--input-file", default=DEFAULT_OUTPUT_FILE, help="Input JSON file")
    check_group.add_argument("-l", "--logs", action="store_true", help="Show detailed logs")
    check_group.add_argument("-n", "--tail-lines", type=int, default=200, help="Log lines to tail")

    args = parser.parse_args()

    # Execute based on mode
    if args.launch:
        launch_tests(args)
    else:  # args.check
        check_tests(args)


if __name__ == "__main__":
    main()
