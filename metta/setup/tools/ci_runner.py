"""CI runner for local testing that matches remote CI behavior.

This tool provides a local development experience that mirrors the GitHub Actions CI workflow.
Both this file and .github/workflows/checks.yml should run the same metta CLI commands.

To verify synchronization, run: uv run python scripts/validate_ci_sync.py
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


def _run_command(cmd: list[str], description: str, *, verbose: bool = False) -> bool:
    """Run a command and return True if successful."""
    info(f"Running: {' '.join(cmd)}")
    try:
        if verbose:
            subprocess.run(cmd, cwd=get_repo_root(), check=True)
        else:
            subprocess.run(
                cmd,
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

    cmd = ["uv", "run", "metta", "lint"]
    passed = _run_command(cmd, "Linting", verbose=verbose)
    return CheckResult("Lint", passed)


def _run_python_tests(*, verbose: bool = False) -> CheckResult:
    """Run Python tests.

    Uses metta pytest command - same command used by CI.
    This ensures local and CI behavior stay perfectly in sync.
    """
    _print_header("Python Tests")

    passed = _run_command(
        ["uv", "run", "metta", "pytest", "--ci"],
        "Python tests",
        verbose=verbose,
    )

    return CheckResult("Python Tests", passed)


def _run_cpp_tests(*, verbose: bool = False) -> CheckResult:
    """Run C++ tests and benchmarks.

    Uses metta cpptest command - same command used by CI.
    This ensures local and CI behavior stay perfectly in sync.
    """
    _print_header("C++ Tests")

    # Run C++ unit tests
    cmd = ["uv", "run", "metta", "cpptest", "--test"]
    if verbose:
        cmd.append("--verbose")
    tests_passed = _run_command(
        cmd,
        "C++ unit tests",
        verbose=verbose,
    )

    if not tests_passed:
        return CheckResult("C++ Tests", False)

    # Run C++ benchmarks
    cmd = ["uv", "run", "metta", "cpptest", "--benchmark"]
    if verbose:
        cmd.append("--verbose")
    benchmarks_passed = _run_command(
        cmd,
        "C++ benchmarks",
        verbose=verbose,
    )
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


def _validate_ci_sync(*, verbose: bool = False) -> bool:
    """Validate that CI runner and workflow are synchronized.

    Returns True if synchronized, False otherwise.
    Prints warning if not synchronized.
    """
    try:
        repo_root = get_repo_root()
        validator_script = repo_root / "scripts" / "validate_ci_sync.py"

        if not validator_script.exists():
            # Silently skip if validator script doesn't exist
            return True

        result = subprocess.run(
            ["uv", "run", "python", str(validator_script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Commands match - silently continue
            return True
        else:
            # Commands don't match - show warning
            console.print()
            console.print("[yellow]⚠ Warning: Local CI runner may be out of sync with GitHub Actions workflow[/yellow]")
            console.print("[yellow]Run 'uv run python scripts/validate_ci_sync.py' for details[/yellow]")
            if verbose:
                console.print()
                console.print(result.stdout)
            console.print()
            return False

    except Exception:
        # Silently skip validation if it fails
        return True


@app.command(name="ci", help="Run CI checks locally")
def cmd_ci(
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error", help="Don't stop on first failure")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
):
    """Run all CI checks locally to match remote CI behavior.

    This runs the same checks that run in CI:
    - Linting (ruff + cpplint)
    - Python tests (metta pytest --ci)
    - C++ tests and benchmarks

    Use 'metta lint' to run only linting checks.
    Use 'metta pytest' to run only Python tests.
    Use 'metta cpptest' to run only C++ tests.
    """
    # Validate synchronization with GitHub Actions workflow (silently passes if OK)
    _validate_ci_sync(verbose=verbose)

    console.print(Panel.fit("[bold]Running CI Checks[/bold]", border_style="cyan"))

    results: list[CheckResult] = []

    # Run linting
    result = _run_lint(verbose=verbose)
    results.append(result)
    if not result.passed and not continue_on_error:
        _print_summary(results)
        error("Linting failed. Fix errors and try again.")
        raise typer.Exit(1)

    # Run Python tests
    result = _run_python_tests(verbose=verbose)
    results.append(result)
    if not result.passed and not continue_on_error:
        _print_summary(results)
        error("Python tests failed. Fix errors and try again.")
        raise typer.Exit(1)

    # Run C++ tests
    result = _run_cpp_tests(verbose=verbose)
    results.append(result)
    if not result.passed and not continue_on_error:
        _print_summary(results)
        error("C++ tests failed. Fix errors and try again.")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
