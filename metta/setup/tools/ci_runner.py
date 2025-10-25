"""CI runner for local testing that matches remote CI behavior.

This tool is the single source of truth for CI checks.
Both local development (metta ci) and GitHub Actions call this same tool.

GitHub Actions workflow calls individual stages:
  - uv run metta ci --stage lint
  - uv run metta ci --stage pytest
  - uv run metta ci --stage cpptest

Local development can run all stages:
  - metta ci (runs all stages)
  - metta ci --stage lint (runs specific stage)
"""

import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

app = typer.Typer(
    help="Run CI checks locally",
    rich_markup_mode="rich",
    no_args_is_help=False,
)

console = Console()


class CheckResult:
    """Result of a CI check."""

    def __init__(self, name: str, passed: bool):
        self.name = name
        self.passed = passed


def _print_header(title: str) -> None:
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 60)


def _run_command(cmd: str, description: str, *, verbose: bool = False) -> bool:
    """Run a shell command and return True if successful."""
    info(f"Running: {cmd}")
    try:
        if verbose:
            subprocess.run(cmd, shell=True, cwd=get_repo_root(), check=True)
        else:
            subprocess.run(
                cmd,
                shell=True,
                cwd=get_repo_root(),
                check=True,
                capture_output=True,
                text=True,
            )
        success(f"{description} passed")
        return True
    except subprocess.CalledProcessError as e:
        error(f"{description} failed")
        if not verbose and hasattr(e, "stdout") and e.stdout:
            console.print(e.stdout)
        if not verbose and hasattr(e, "stderr") and e.stderr:
            console.print(e.stderr)
        return False


def _run_lint(*, verbose: bool = False) -> CheckResult:
    """Run linting checks (Python and C++).

    Uses metta lint command - same command used by CI.
    This ensures local and CI behavior stay perfectly in sync.
    """
    _print_header("Linting")

    cmd = "uv run metta lint"
    passed = _run_command(cmd, "Linting", verbose=verbose)
    return CheckResult("Lint", passed)


def _run_python_tests(*, verbose: bool = False, extra_args: list[str] | None = None) -> CheckResult:
    """Run Python tests.

    Uses metta pytest command - same command used by CI.
    This ensures local and CI behavior stay perfectly in sync.
    """
    _print_header("Python Tests")

    cmd = "uv run metta pytest --ci"
    if extra_args:
        cmd += " " + " ".join(extra_args)
    passed = _run_command(cmd, "Python tests", verbose=verbose)

    return CheckResult("Python Tests", passed)


def _run_cpp_tests(*, verbose: bool = False) -> CheckResult:
    """Run C++ tests and benchmarks.

    Uses metta cpptest command - same command used by CI.
    This ensures local and CI behavior stay perfectly in sync.
    """
    _print_header("C++ Tests")

    # Run C++ unit tests
    cmd = "uv run metta cpptest --test --verbose"
    tests_passed = _run_command(cmd, "C++ unit tests", verbose=verbose)

    if not tests_passed:
        return CheckResult("C++ Tests", False)

    # Run C++ benchmarks
    cmd = "uv run metta cpptest --benchmark --verbose"
    benchmarks_passed = _run_command(cmd, "C++ benchmarks", verbose=verbose)
    if not benchmarks_passed:
        return CheckResult("C++ Tests", False)

    return CheckResult("C++ Tests", True)


def _print_summary(results: list[CheckResult]) -> None:
    """Print a summary table of check results."""
    console.print()

    table = Table(title="CI Check Summary", show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")

    for result in results:
        status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
        table.add_row(result.name, status)

    console.print(table)
    console.print()


@app.command(
    name="ci", help="Run CI checks locally", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def cmd_ci(
    ctx: typer.Context,
    stage: Annotated[str | None, typer.Option(help="Run specific stage: lint, pytest, or cpptest")] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error", help="Don't stop on first failure")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Run CI checks locally to match remote CI behavior.

    This tool is the single source of truth for CI checks.
    GitHub Actions and local development both use this command.

    Examples:
        metta ci                                    # Run all stages (local development)
        metta ci --stage lint                       # Run only linting (used by CI)
        metta ci --stage pytest                     # Run only Python tests (used by CI)
        metta ci --stage pytest --skip-package app  # Pass extra args to pytest
        metta ci --stage cpptest                    # Run only C++ tests (used by CI)

    Individual tools can also be run directly:
        metta lint          # Run only linting checks
        metta pytest        # Run only Python tests
        metta cpptest       # Run only C++ tests
    """
    # Get any extra arguments passed after known options
    extra_args = ctx.args if hasattr(ctx, "args") else []

    # Map of valid stages to their runner functions
    stages = {
        "lint": lambda v: _run_lint(verbose=v),
        "pytest": lambda v: _run_python_tests(verbose=v, extra_args=extra_args),
        "cpptest": lambda v: _run_cpp_tests(verbose=v),
    }

    # If specific stage requested, run only that stage
    if stage:
        if stage not in stages:
            error(f"Unknown stage: {stage}")
            info(f"Valid stages: {', '.join(stages.keys())}")
            raise typer.Exit(1)

        # Run the specific stage
        result = stages[stage](verbose)
        if result.passed:
            success(f"Stage '{stage}' passed!")
            sys.exit(0)
        else:
            error(f"Stage '{stage}' failed.")
            sys.exit(1)

    # Otherwise run all stages (local development workflow)
    console.print(Panel.fit("[bold]Running All CI Checks[/bold]", border_style="cyan"))

    results: list[CheckResult] = []

    # Run all stages in order
    for stage_name, stage_func in stages.items():
        result = stage_func(verbose)
        results.append(result)
        if not result.passed and not continue_on_error:
            _print_summary(results)
            error(f"Stage '{stage_name}' failed. Fix errors and try again.")
            raise typer.Exit(1)

    # Print summary
    _print_summary(results)

    # Determine overall status
    all_passed = all(r.passed for r in results)
    if all_passed:
        success("All CI checks passed!")
        sys.exit(0)
    else:
        error("Some CI checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    app()
