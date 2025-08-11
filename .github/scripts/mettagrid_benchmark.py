#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Build mettagrid C++ benchmarks and check for compiler warnings and errors.
This script is designed to be run in CI to surface build issues.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.mettagrid_build_utils import (
    BuildChecker,
    run_build_command,
    setup_build_environment,
    write_github_outputs,
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

    # skip GitHub Actions summary since it largely duplicated the test step
    # github_summary = checker.generate_github_summary(title="Benchmark Summary")
    # write_github_summary(github_summary)

    # Set outputs for GitHub Actions
    write_github_outputs(checker, build_success, build_output)

    # Exit with appropriate code - only fail on actual build failure or errors
    if not build_success:
        print("\n❌ Build command failed!")
        sys.exit(1)
    elif checker.build_failed:
        print("\n❌ Build completed with errors!")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
