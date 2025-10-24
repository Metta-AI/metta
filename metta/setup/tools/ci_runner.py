"""CI runner for local testing that matches remote CI behavior.

This tool provides a local development experience that mirrors the GitHub Actions CI workflow.
Both this file and .github/workflows/checks.yml should run the same metta CLI commands.

Validation is performed automatically (and silently) when running `metta ci`.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
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


def _extract_commands_from_python(file_path: Path) -> dict[str, list[str]]:
    """Extract metta commands from this ci_runner.py file.

    Returns dict mapping check name to list of command strings.
    """
    content = file_path.read_text()
    tree = ast.parse(content)
    commands = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_run_"):
            check_name = node.name.replace("_run_", "")
            check_commands = []

            for child in ast.walk(node):
                if isinstance(child, ast.List):
                    cmd_parts = [str(elt.value) for elt in child.elts if isinstance(elt, ast.Constant)]
                    if cmd_parts and (cmd_parts[0] in ("uv", "metta") or cmd_parts[0:2] == ["uv", "run"]):
                        normalized = [p for p in cmd_parts if p not in ("--verbose", "-v")]
                        check_commands.append(" ".join(normalized))

            if check_commands:
                commands[check_name] = check_commands

    return commands


def _extract_commands_from_yaml(file_path: Path) -> dict[str, list[str]]:
    """Extract metta commands from checks.yml workflow.

    Returns dict mapping job name to list of command strings.
    """
    content = file_path.read_text()
    data = yaml.safe_load(content)
    commands = {}

    if "jobs" not in data:
        return commands

    for job_name, job_data in data["jobs"].items():
        if not isinstance(job_data, dict) or "steps" not in job_data:
            continue

        job_commands = []
        for step in job_data["steps"]:
            if not isinstance(step, dict):
                continue

            run_cmd = step.get("run", "")
            if not run_cmd:
                continue

            for line in run_cmd.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "metta" in line and ("uv run" in line or line.startswith("metta")):
                    normalized = line
                    normalized = re.sub(r"\$\{.*?\}|\$\(.*?\)", "", normalized)
                    normalized = normalized.replace('"', "").replace("'", "")
                    normalized = re.sub(r"\s+--verbose\b", "", normalized)
                    normalized = re.sub(r"\s+\|\|.*$", "", normalized)
                    normalized = re.sub(r"\s+&&.*$", "", normalized)
                    normalized = " ".join(normalized.split())

                    if normalized:
                        job_commands.append(normalized)

        if job_commands:
            commands[job_name] = job_commands

    return commands


def _check_ci_sync_silent() -> bool:
    """Silently check if CI runner and workflow are synchronized.

    Returns True if synchronized, False otherwise.
    """
    try:
        repo_root = get_repo_root()
        python_file = repo_root / "metta/setup/tools/ci_runner.py"
        yaml_file = repo_root / ".github/workflows/checks.yml"

        if not python_file.exists() or not yaml_file.exists():
            return True

        python_cmds = _extract_commands_from_python(python_file)
        yaml_cmds = _extract_commands_from_yaml(yaml_file)

        # Build command sets
        python_cmd_set = {" ".join(cmd.split()) for cmds in python_cmds.values() for cmd in cmds}
        yaml_cmd_set = {" ".join(cmd.split()) for cmds in yaml_cmds.values() for cmd in cmds}

        return python_cmd_set == yaml_cmd_set

    except Exception:
        return True  # Silently pass on errors


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
    app()


if __name__ == "__main__":
    main()
