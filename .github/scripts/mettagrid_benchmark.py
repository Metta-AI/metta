#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Build mettagrid C++ benchmarks and check for compiler warnings and errors.
This script is designed to be run in CI to surface build issues.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.mettagrid_build_utils import (
    BuildChecker,
    run_build_command,
    setup_build_environment,
    write_github_outputs,
    write_github_summary,
)


def run_benchmark_build(project_root: Path) -> tuple[bool, str]:
    """Run the benchmark build process and capture output."""
    env = setup_build_environment(project_root)

    print("🔨 Building benchmarks...")

    # Activate venv and build benchmarks
    build_cmd = ["bash", "-c", "source ../.venv/bin/activate && make benchmark VERBOSE=1"]

    return run_build_command(build_cmd, project_root, env)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check mettagrid benchmark build for compiler warnings and errors")
    parser.add_argument(
        "-w", "--max-warnings", type=int, default=50, help="Maximum allowed warnings before failing (default: 50)"
    )

    args = parser.parse_args()

    # Determine project root (assumes script is in .github/scripts/)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent

    # The mettagrid subproject is in the mettagrid directory
    project_root = repo_root / "mettagrid"

    print(f"Repository root: {repo_root}")
    print(f"Mettagrid project root: {project_root}")

    # Check if we're in the right directory
    if not (project_root / "Makefile").exists():
        print(f"Error: No Makefile found in {project_root}")
        print(f"Looking for: {project_root / 'Makefile'}")
        sys.exit(1)

    # Run the build
    build_success, build_output = run_benchmark_build(project_root)

    # Analyze the build output
    checker = BuildChecker(project_root)
    checker.parse_build_output(build_output)

    # Print summary to console
    checker.print_summary()

    # Check build quality
    print(f"\n📊 Benchmark Build Quality Check (max warnings: {args.max_warnings})")
    print("=" * 50)

    exit_code = 0

    if not build_success:
        print("❌ Benchmark build command failed!")
        exit_code = 1
    elif checker.build_failed:
        print("❌ Benchmark build completed with errors!")
        exit_code = 1
    elif checker.total_warnings > args.max_warnings:
        print(f"❌ Too many warnings! ({checker.total_warnings} > {args.max_warnings})")
        exit_code = 1
    elif checker.runtime_issues > 0:
        print(f"❌ Runtime issues detected: {checker.runtime_issues}")
        exit_code = 1
    else:
        print("✅ Benchmark build quality check passed")

    print("=" * 50)

    # Generate and write GitHub Actions summary if needed
    if exit_code != 0:
        # Only write summary on failure to avoid duplication with test step
        github_summary = checker.generate_github_summary(title="Benchmark Build Summary")

        # Add quality check result to summary
        github_summary += "\n\n### ❌ Benchmark Build Quality Check Failed\n"
        if not build_success:
            github_summary += "- Build command failed\n"
        if checker.build_failed:
            github_summary += f"- Build errors: {checker.total_errors}\n"
        if checker.total_warnings > args.max_warnings:
            github_summary += f"- Too many warnings: {checker.total_warnings} > {args.max_warnings}\n"
        if checker.runtime_issues > 0:
            github_summary += f"- Runtime issues: {checker.runtime_issues}\n"

        write_github_summary(github_summary)

    # Set outputs for GitHub Actions
    write_github_outputs(checker, build_success, build_output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
