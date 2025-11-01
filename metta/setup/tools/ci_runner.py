"""CI runner for local testing that matches remote CI behavior.

This tool is the single source of truth for CI checks.
Both local development (metta ci) and GitHub Actions call this same tool.

GitHub Actions workflow calls individual stages:
  - uv run metta ci --stage lint
  - uv run metta ci --stage python-tests
  - uv run metta ci --stage cpp-tests
  - uv run metta ci --stage cpp-benchmarks

Local development can run all stages:
  - metta ci (runs all stages)
  - metta ci --stage <name> (runs specific stage)
"""

import shlex
import subprocess
import sys
from typing import Annotated, Callable, Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from metta.common.util.fs import get_repo_root
from metta.setup.tools.test_runner.test_python import PACKAGES as PYTEST_PACKAGES
from metta.setup.utils import error, info, success

console = Console()

# Allow skipping any package supported by metta pytest runner.
ALLOWED_SKIP_PACKAGES = {package.name.lower() for package in PYTEST_PACKAGES}


class CheckResult:
    """Result of a CI check."""

    def __init__(self, name: str, passed: bool):
        self.name = name
        self.passed = passed


def _format_cmd_for_display(cmd: Sequence[str]) -> str:
    return shlex.join(cmd)


def _print_header(title: str) -> None:
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 60)


def _ensure_no_extra_args(stage_name: str, extra_args: Sequence[str] | None) -> None:
    if extra_args:
        error(f"Stage '{stage_name}' does not accept extra arguments.")
        raise typer.Exit(1)


def _normalize_python_stage_args(extra_args: Sequence[str] | None) -> list[str]:
    if not extra_args:
        return []

    sanitized: list[str] = []
    args = list(extra_args)
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--skip-package":
            if idx + 1 >= len(args):
                error("'--skip-package' requires a package name.")
                raise typer.Exit(1)
            package_name = args[idx + 1]
            if package_name.lower() not in ALLOWED_SKIP_PACKAGES:
                allowed = ", ".join(sorted(ALLOWED_SKIP_PACKAGES))
                error(f"Unsupported package '{package_name}' for --skip-package.")
                info(f"Allowed packages: {allowed}")
                raise typer.Exit(1)
            sanitized.extend([token, package_name])
            idx += 2
            continue

        error(f"Argument '{token}' is not supported for python-tests stage.")
        info("Allowed arguments: --skip-package <package>")
        raise typer.Exit(1)

    return sanitized


def _run_command(cmd: Sequence[str], description: str, *, verbose: bool = False) -> bool:
    """Run a command and return True if successful."""
    display_cmd = _format_cmd_for_display(cmd)
    info(f"Running: {display_cmd}")
    try:
        subprocess.run(
            cmd,
            cwd=get_repo_root(),
            check=True,
            capture_output=not verbose,
            text=True,
        )
        success(f"{description} passed")
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - invoked via subprocess
        error(f"{description} failed")
        if not verbose:
            if exc.stdout:
                console.print(exc.stdout, markup=False)
            if exc.stderr:
                console.print(exc.stderr, markup=False)
        return False


def _run_lint(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Run linting checks (Python and C++)."""
    _ensure_no_extra_args("lint", extra_args)
    _print_header("Linting")

    cmd = ["uv", "run", "metta", "lint", "--check"]
    passed = _run_command(cmd, "Linting", verbose=verbose)
    return CheckResult("Lint", passed)


def _run_python_tests(
    *,
    verbose: bool = False,
    extra_args: Sequence[str] | None = None,
) -> CheckResult:
    """Run Python tests (excludes benchmarks)."""
    _print_header("Python Tests")

    cmd = ["uv", "run", "metta", "pytest", "--ci"]
    cmd.extend(_normalize_python_stage_args(extra_args))
    passed = _run_command(cmd, "Python tests", verbose=verbose)

    return CheckResult("Python Tests", passed)


def _run_cpp_tests(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Run C++ unit tests (excludes benchmarks)."""
    _ensure_no_extra_args("cpp-tests", extra_args)
    _print_header("C++ Tests")

    cmd = ["uv", "run", "metta", "cpptest", "--test"]
    if verbose:
        cmd.append("--verbose")
    passed = _run_command(cmd, "C++ unit tests", verbose=verbose)

    return CheckResult("C++ Tests", passed)


def _run_cpp_benchmarks(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Run C++ benchmarks."""
    _ensure_no_extra_args("cpp-benchmarks", extra_args)
    _print_header("C++ Benchmarks")

    cmd = ["uv", "run", "metta", "cpptest", "--benchmark"]
    if verbose:
        cmd.append("--verbose")
    passed = _run_command(cmd, "C++ benchmarks", verbose=verbose)

    return CheckResult("C++ Benchmarks", passed)


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


StageRunner = Callable[[bool, Sequence[str] | None], CheckResult]

stages: dict[str, StageRunner] = {
    "lint": lambda v, args: _run_lint(verbose=v, extra_args=args),
    "python-tests": lambda v, args: _run_python_tests(verbose=v, extra_args=args),
    "cpp-tests": lambda v, args: _run_cpp_tests(verbose=v, extra_args=args),
    "cpp-benchmarks": lambda v, args: _run_cpp_benchmarks(verbose=v, extra_args=args),
}


def cmd_ci(
    ctx: typer.Context,
    stage: Annotated[
        str | None,
        typer.Option(help=f"Run specific stage: {', '.join(stages.keys())}"),
    ] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error", help="Don't stop on first failure")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Run CI checks locally to match remote CI behavior."""
    extra_args = list(getattr(ctx, "args", []))

    if extra_args and stage is None:
        error("Extra arguments require specifying a --stage.")
        raise typer.Exit(1)

    # If specific stage requested, run only that stage
    if stage:
        if stage not in stages:
            error(f"Unknown stage: {stage}")
            info(f"Valid stages: {', '.join(stages.keys())}")
            raise typer.Exit(1)

        # Run the specific stage
        result = stages[stage](verbose, extra_args)
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
        result = stage_func(verbose, None)
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


# No main - this module exports cmd_ci for use in metta_cli.py
