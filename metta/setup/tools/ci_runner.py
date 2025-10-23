"""CI runner for local testing that matches remote CI behavior."""

import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success, warning

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
    """Run linting checks (Python and C++)."""
    _print_header("Linting")

    # Run Python linting
    info("Running Python linting...")
    ruff_format_passed = _run_command(
        ["uv", "run", "ruff", "format", "--check", "."],
        "Ruff format check",
        verbose=verbose,
    )

    if not ruff_format_passed:
        return CheckResult("Lint", False)

    ruff_check_passed = _run_command(
        ["uv", "run", "ruff", "check", "--exit-non-zero-on-fix", "."],
        "Ruff check",
        verbose=verbose,
    )

    if not ruff_check_passed:
        return CheckResult("Lint", False)

    # Run C++ linting
    info("Running C++ linting...")
    cpplint_script = get_repo_root() / "packages" / "mettagrid" / "tests" / "cpplint.sh"
    if cpplint_script.exists():
        cpplint_passed = _run_command(
            ["bash", str(cpplint_script)],
            "C++ lint",
            verbose=verbose,
        )
        if not cpplint_passed:
            return CheckResult("Lint", False)
    else:
        warning("C++ linting script not found, skipping")

    return CheckResult("Lint", True)


def _run_python_tests(*, verbose: bool = False) -> CheckResult:
    """Run Python tests."""
    _print_header("Python Tests")

    passed = _run_command(
        ["uv", "run", "metta", "pytest", "--ci"],
        "Python tests",
        verbose=verbose,
    )

    return CheckResult("Python Tests", passed)


def _run_cpp_tests(*, verbose: bool = False) -> CheckResult:
    """Run C++ tests and benchmarks."""
    _print_header("C++ Tests")

    # Run C++ unit tests
    tests_passed = _run_command(
        ["uv", "run", "metta", "cpptest", "--test", "--verbose"],
        "C++ unit tests",
        verbose=verbose,
    )

    if not tests_passed:
        return CheckResult("C++ Tests", False)

    # Run C++ benchmarks
    benchmarks_passed = _run_command(
        ["uv", "run", "metta", "cpptest", "--benchmark", "--verbose"],
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
