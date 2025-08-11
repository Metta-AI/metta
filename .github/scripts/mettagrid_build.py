#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Build mettagrid C++ project and check for compiler warnings and errors.
This script is designed to be run in CI to surface build issues.
"""

import argparse
import subprocess
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


def run_build(project_root: Path, with_coverage: bool = False) -> tuple[bool, str]:
    """Run the build process and capture output."""
    env = setup_build_environment(project_root)

    print("🧹 Cleaning build artifacts...")
    clean_result = subprocess.run(["make", "clean"], cwd=project_root, capture_output=True, text=True, env=env)

    if clean_result.returncode != 0:
        print(f"Warning: 'make clean' failed: {clean_result.stderr}")

    # Choose build target based on coverage requirement
    if with_coverage:
        print("🔨 Building project with coverage...")
        build_target = "coverage"
    else:
        print("🔨 Building project...")
        build_target = "build"

    # Force verbose output to capture compiler messages
    build_cmd = ["make", build_target, "VERBOSE=1"]

    return run_build_command(build_cmd, project_root, env)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check mettagrid build for compiler warnings and errors")
    parser.add_argument("-c", "--with-coverage", action="store_true", help="Build with coverage enabled")
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
    build_success, build_output = run_build(project_root, args.with_coverage)

    # Analyze the build output
    checker = BuildChecker(project_root)
    checker.parse_build_output(build_output)

    # Print summary to console
    checker.print_summary()

    # Check build quality
    print(f"\n📊 Build Quality Check (max warnings: {args.max_warnings})")
    print("=" * 50)

    exit_code = 0

    if not build_success:
        print("❌ Build command failed!")
        exit_code = 1
    elif checker.build_failed:
        print("❌ Build completed with errors!")
        exit_code = 1
    elif checker.total_warnings > args.max_warnings:
        print(f"❌ Too many warnings! ({checker.total_warnings} > {args.max_warnings})")
        exit_code = 1
    elif checker.runtime_issues > 0:
        print(f"❌ Runtime issues detected: {checker.runtime_issues}")
        exit_code = 1
    else:
        print("✅ Build quality check passed")

    print("=" * 50)

    # Generate and write GitHub Actions summary
    github_summary = checker.generate_github_summary()

    # Add quality check result to summary
    if exit_code != 0:
        github_summary += "\n\n### ❌ Build Quality Check Failed\n"
        if not build_success:
            github_summary += "- Build command failed\n"
        if checker.build_failed:
            github_summary += f"- Build errors: {checker.total_errors}\n"
        if checker.total_warnings > args.max_warnings:
            github_summary += f"- Too many warnings: {checker.total_warnings} > {args.max_warnings}\n"
        if checker.runtime_issues > 0:
            github_summary += f"- Runtime issues: {checker.runtime_issues}\n"
    else:
        github_summary += "\n\n### ✅ Build Quality Check Passed\n"
        github_summary += f"- Warnings: {checker.total_warnings}/{args.max_warnings}\n"

    write_github_summary(github_summary)

    # Set outputs for GitHub Actions
    write_github_outputs(checker, build_success, build_output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
