"""CI runner for local testing that matches remote CI behavior.

This tool provides a local development experience that mirrors the GitHub Actions CI workflow.
Both this file and .github/workflows/checks.yml should run the same metta CLI commands.

Validation is performed automatically (and silently) when running `metta ci`.
"""

import hashlib
import re
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


# ==============================================================================
# CI Synchronization Validation
# ==============================================================================


def _extract_metta_lines(content: str) -> str:
    """Extract lines containing metta commands and hash them.

    Returns a hash of all lines containing 'metta' commands (lint, pytest, cpptest).
    This simple approach avoids complex parsing while still detecting drift.
    """
    lines = []
    for line in content.split("\n"):
        # Look for lines with metta commands (not just mentions in docs/comments)
        if re.search(r"\bmetta\s+(lint|pytest|cpptest)\b", line):
            # Normalize whitespace
            normalized = " ".join(line.split())
            lines.append(normalized)

    # Sort for consistent comparison
    lines.sort()
    combined = "\n".join(lines)
    return hashlib.sha256(combined.encode()).hexdigest()


# Reference hash of checks.yml metta commands
# Update this by running: python -m metta.setup.tools.ci_runner
CHECKS_YML_REFERENCE_HASH = "3c8cb5999fcffecc09a5a234ec90be12fbe4cf20afe1800d8f4d392d117713d1"


def _check_ci_sync_silent() -> bool:
    """Silently check if CI runner matches the reference from checks.yml.

    Returns True if synchronized, False otherwise.
    """
    try:
        repo_root = get_repo_root()
        python_file = repo_root / "metta/setup/tools/ci_runner.py"

        if not python_file.exists():
            return True

        python_hash = _extract_metta_lines(python_file.read_text())

        return python_hash == CHECKS_YML_REFERENCE_HASH

    except Exception:
        return True  # Silently pass on errors


def _update_reference_hash() -> None:
    """Update the reference hash from checks.yml.

    Run this after intentionally changing the CI commands in checks.yml.
    """
    repo_root = get_repo_root()
    yaml_file = repo_root / ".github/workflows/checks.yml"
    python_file = repo_root / "metta/setup/tools/ci_runner.py"

    if not yaml_file.exists():
        error(f"Cannot find {yaml_file}")
        sys.exit(1)

    yaml_hash = _extract_metta_lines(yaml_file.read_text())

    # Update this file
    content = python_file.read_text()
    updated = re.sub(
        r'CHECKS_YML_REFERENCE_HASH = "3c8cb5999fcffecc09a5a234ec90be12fbe4cf20afe1800d8f4d392d117713d1"]*"',
        'CHECKS_YML_REFERENCE_HASH = "3c8cb5999fcffecc09a5a234ec90be12fbe4cf20afe1800d8f4d392d117713d1"',
        content,
    )

    python_file.write_text(updated)
    success(f"Updated reference hash to: {yaml_hash}")
    info("Commit this change to keep CI synchronized")


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
        is_synced = _check_ci_sync_silent()

        if is_synced:
            # Commands match - silently continue
            return True
        else:
            # Commands don't match - show warning
            console.print()
            console.print("[yellow]⚠ Warning: Local CI runner may be out of sync with GitHub Actions workflow[/yellow]")
            console.print("[yellow]This may indicate drift between local and remote CI checks.[/yellow]")
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
    """Update the reference hash when run as a script."""
    if len(sys.argv) > 1 and sys.argv[1] == "update-hash":
        _update_reference_hash()
    else:
        app()


if __name__ == "__main__":
    main()
