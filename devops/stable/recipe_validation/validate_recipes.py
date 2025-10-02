#!/usr/bin/env -S uv run
"""CLI for validating recipes across tools.

Usage:
    validate_recipes.py launch <tool>     # Launch validations for a specific tool
    validate_recipes.py check              # Check status of launched validations
    validate_recipes.py check -l           # Check with detailed logs
"""

import argparse
import sys
from pathlib import Path

from devops.skypilot.utils.testing_helpers import SkyPilotJobChecker
from devops.stable.recipe_validation.recipe_validator import (
    TOOL_VALIDATIONS,
    RecipeValidator,
    ValidationLocation,
)
from devops.stable.state_manager import StateManager
from metta.common.util.text_styles import bold, cyan, green, red, yellow


def launch_tool_validations(args):
    """Launch validations for a specific tool."""
    tool_name = args.tool

    if tool_name not in TOOL_VALIDATIONS:
        print(red(f"Unknown tool: {tool_name}"))
        print(f"Available tools: {', '.join(TOOL_VALIDATIONS.keys())}")
        sys.exit(1)

    tool = TOOL_VALIDATIONS[tool_name]

    print(f"\n{bold(f'=== Validating {tool.name} ===')}")
    print(f"{cyan('Description:')} {tool.description}")
    print(f"{cyan('Recipes to validate:')} {len(tool.recipes)}")
    print()

    # Show what will be validated
    for recipe in tool.recipes:
        location_str = "🏠 Local" if recipe.location == ValidationLocation.LOCAL else "☁️  Remote"
        print(f"  • {yellow(recipe.name)} [{location_str}]: {recipe.description}")
    print()

    # Initialize state manager
    state_manager = StateManager()

    # Create or load state
    version = args.version or f"validation_{tool_name}"
    state = state_manager.load_state(version)
    if state is None:
        repo_root = RecipeValidator._get_repo_root()
        state = state_manager.create_state(version, repo_root)
        print(f"{cyan('Created new validation state:')} {version}")
    else:
        print(f"{cyan('Resuming validation state:')} {version}")

    # Create validator
    validator = RecipeValidator(
        base_name=f"{tool_name}_validation",
        skip_git_check=args.skip_git_check,
        state_manager=state_manager,
    )

    # Check git state for remote validations
    if any(r.location == ValidationLocation.REMOTE for r in tool.recipes):
        launcher = validator._ensure_remote_launcher()
        if not launcher.check_git_state():
            sys.exit(1)

    # Run validations
    results = validator.validate_tool(tool)

    # Save validation results to state
    for _name, check_result in validator.validation_results.items():
        state.add_validation_result(check_result)

    # Save state
    state_path = state_manager.save_state(state)
    print(f"\n{cyan('State saved to:')} {state_path}")

    # Save results if remote launcher was used
    if validator.remote_launcher:
        output_file = args.output_file or f"{tool_name}_validation_jobs.json"
        output_path = validator.remote_launcher.save_results(output_file)
        print(f"{cyan('Remote job tracker saved to:')} {output_path.absolute()}")
        validator.remote_launcher.print_summary()

        # Exit with error if any launches failed
        if validator.remote_launcher.failed_launches:
            sys.exit(1)

    # Print overall results
    print(f"\n{bold('Validation Results:')}")
    status_colors = {"passed": green, "failed": red, "running": cyan, "pending": yellow, "skipped": yellow}
    for recipe_name, status in results.items():
        color_fn = status_colors.get(status.value, yellow)
        print(f"  {recipe_name}: {color_fn(status.value)}")

    # Print summary
    summary = state.validation_summary
    print(f"\n{bold('Summary:')}")
    if summary["passed"] > 0:
        print(f"  {green('Passed')}: {summary['passed']}")
    if summary["failed"] > 0:
        print(f"  {red('Failed')}: {summary['failed']}")
    if summary["running"] > 0:
        print(f"  {cyan('Running')}: {summary['running']}")

    # Exit with error if any validations failed
    if any(status == "failed" for status in results.values()):
        sys.exit(1)


def check_validations(args):
    """Check status of launched validations."""
    # Determine input file
    input_file = args.input_file
    if not input_file:
        # Try to find any *_validation_jobs.json file
        json_files = list(Path(".").glob("*_validation_jobs.json"))
        if not json_files:
            print(red("No validation job files found"))
            print("Run 'validate_recipes.py launch <tool>' first")
            sys.exit(1)
        elif len(json_files) > 1:
            print(yellow("Multiple validation job files found:"))
            for f in json_files:
                print(f"  • {f}")
            print("\nSpecify which file to check with -f/--input-file")
            sys.exit(1)
        else:
            input_file = str(json_files[0])

    # Create checker
    checker = SkyPilotJobChecker(input_file=input_file)

    # Load jobs
    if not checker.load_jobs():
        print("Run 'validate_recipes.py launch <tool>' first to create the job file")
        sys.exit(1)

    # Get job count
    launched_jobs = checker.jobs_data.get("launched_jobs", [])
    if not launched_jobs:
        sys.exit(0)

    # Summary header
    validation_info = checker.jobs_data.get("test_run_info", {})
    print(bold(f"\n=== Checking {len(launched_jobs)} Recipe Validation Jobs ==="))
    print(f"{cyan('Validation run:')} {validation_info.get('base_name', 'Unknown')}")
    print(f"{cyan('Launch time:')} {validation_info.get('launch_time', 'Unknown')}")
    print(f"{cyan('Input file:')} {input_file}")

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
    print(f"  • Use {cyan('check -l')} to view detailed job logs")
    print(f"  • Use {cyan('check -n <lines>')} to change log lines to tail")
    print(f"  • Use {cyan('sky jobs logs <job_id>')} to view a single job's full log")


def main():
    parser = argparse.ArgumentParser(
        description="Recipe validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Launch subcommand
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch recipe validations for a tool",
        description="Launch recipe validations for a specific tool",
    )
    launch_parser.add_argument(
        "tool",
        choices=list(TOOL_VALIDATIONS.keys()),
        help="Tool to validate",
    )
    launch_parser.add_argument(
        "--version", help="Version/state name (default: auto-generated from tool name)", default=None
    )
    launch_parser.add_argument("--output-file", help="Output JSON file (default: <tool>_validation_jobs.json)")
    launch_parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")

    # Check subcommand
    check_parser = subparsers.add_parser(
        "check",
        help="Check validation results",
        description="Check the status and results of launched validations",
    )
    check_parser.add_argument("-f", "--input-file", help="Input JSON file (auto-detected if not specified)")
    check_parser.add_argument("-l", "--logs", action="store_true", help="Show detailed logs")
    check_parser.add_argument("-n", "--tail-lines", type=int, default=200, help="Log lines to tail")

    args = parser.parse_args()

    if args.command == "launch":
        launch_tool_validations(args)
    elif args.command == "check":
        check_validations(args)


if __name__ == "__main__":
    main()
