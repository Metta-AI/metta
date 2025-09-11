#!/usr/bin/env -S uv run
"""
Test script for validating recipes.

Usage:
    recipe_test.py --launch    # Launch test jobs
    recipe_test.py --check     # Check test results
    recipe_test.py --check -l  # Check with detailed logs
"""

import argparse
import sys

from devops.skypilot.utils.testing_helpers import SkyPilotJobChecker, SkyPilotTestLauncher, TestCondition
from metta.common.util.text_styles import bold, cyan, yellow

# Recipe configurations
RECIPES = {
    "arena_basic_easy_shaped": {
        "module": "experiments.recipes.arena_basic_easy_shaped.train",
        "description": "Basic arena with easy shaping",
    },
    "arena": {
        "module": "experiments.recipes.arena.train",
        "description": "Standard arena recipe",
    },
    "icl_resource_chain": {
        "module": "experiments.recipes.icl_resource_chain.train",
        "description": "In-context learning resource chain",
    },
    "navigation": {
        "module": "experiments.recipes.navigation.train",
        "description": "Navigation task",
    },
    "navigation_sequence": {
        "module": "experiments.recipes.navigation_sequence.train",
        "description": "Sequential navigation task",
    },
}

# Test condition - normal completion with short training
TEST_CONDITION = TestCondition(
    name="Normal Completion",
    extra_args=["--overrides", "trainer.total_timesteps=50000"],
    description="Exit normally after 50k timesteps",
    ci=False,
)

# Base configuration
BASE_ARGS = ["--no-spot", "--gpus=4", "--nodes", "1"]

# Default file name
DEFAULT_OUTPUT_FILE = "recipe_test_jobs.json"


def launch_tests(args):
    """Launch recipe test jobs."""
    # Create launcher
    launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

    # Check git state
    if not launcher.check_git_state():
        sys.exit(1)

    # Show test configuration
    print(f"\n{bold('=== Recipe Test Configuration ===')}")
    print(f"{cyan('Recipes to test:')}")
    for recipe_key, recipe in RECIPES.items():
        print(f"  • {yellow(recipe_key)}: {recipe['description']}")
    print(f"\n{cyan('Test condition:')} {TEST_CONDITION.description}")
    print(f"{cyan('Nodes:')} 1")
    print(f"{cyan('CI tests:')} Disabled")
    print(f"\n{cyan('Total jobs to launch:')} {len(RECIPES)}")
    print(f"{cyan('Output file:')} {args.output_file}")

    # Launch jobs
    for recipe_key, recipe in RECIPES.items():
        # Generate run name
        run_name = launcher.generate_run_name(recipe_key)

        # Test config for tracking
        test_config = {
            "recipe": recipe_key,
            "description": recipe["description"],
            "timesteps": 50000,
            "nodes": 1,
            "ci_tests_enabled": False,
        }

        # Launch the job
        launcher.launch_job(
            module=recipe["module"],
            run_name=run_name,
            base_args=BASE_ARGS,
            extra_args=TEST_CONDITION.extra_args,
            test_config=test_config,
            enable_ci_tests=False,
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
    """Check recipe test results."""
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
    print(bold(f"\n=== Checking {len(launched_jobs)} Recipe Test Jobs ==="))
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
        description="Recipe test launcher and checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --launch                    # Launch recipe test jobs
  %(prog)s --check                     # Check job results
  %(prog)s --check -l                  # Check with detailed logs
  %(prog)s --launch --skip-git-check   # Launch without git validation
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--launch", action="store_true", help="Launch test jobs")
    mode_group.add_argument("--check", action="store_true", help="Check test results")

    # Launch options
    launch_group = parser.add_argument_group("launch options")
    launch_group.add_argument("--base-name", default="recipe_test", help="Base name for test runs")
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
