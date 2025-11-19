"""CI runner for local testing that matches remote CI behavior.

This tool is the single source of truth for CI checks.
Both local development (metta ci) and GitHub Actions call this same tool.

GitHub Actions workflow calls individual stages:
  - uv run metta ci --stage lint
  - uv run metta ci --stage python-tests-and-benchmarks  # includes Pyright
  - uv run metta ci --stage cpp-tests
  - uv run metta ci --stage cpp-benchmarks
  - uv run metta ci --stage recipe-tests

Local development can run all stages:
  - metta ci (runs all stages)
  - metta ci --stage <name> (runs specific stage)
"""

import logging
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated, Callable, Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from metta.common.util.fs import get_repo_root
from metta.jobs.job_api import submit_monitor_and_report
from metta.jobs.job_manager import JobManager
from metta.setup.tools.test_runner.test_python import PACKAGES as PYTEST_PACKAGES
from metta.setup.utils import error, info, success
from recipes.validation.ci_suite import get_ci_jobs

console = Console()

# Allow skipping any package supported by metta pytest runner.
ALLOWED_SKIP_PACKAGES = {package.name.lower() for package in PYTEST_PACKAGES}

# Always run Pyright on the shared metta package plus any opted-in workspace packages.
PYRIGHT_BASE_TARGETS: tuple[str, ...] = ("metta",)
PYRIGHT_PACKAGE_TARGETS: dict[str, str] = {
    "agent": "agent/src/metta/agent",
    "app_backend": "app_backend/src/metta/app_backend",
    "common": "common/src/metta/common",
    "cogames": "packages/cogames/src/cogames",
    "cortex": "packages/cortex/src/cortex",
    "mettagrid": "packages/mettagrid/python/src/mettagrid",
}
PYRIGHT_ENFORCE_ENV = "METTA_CI_ENFORCE_PYRIGHT"


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


def _extract_skip_packages(normalized_args: Sequence[str]) -> set[str]:
    """Collect packages that the caller explicitly skipped."""
    skipped: set[str] = set()
    idx = 0
    while idx < len(normalized_args):
        token = normalized_args[idx]
        if token == "--skip-package" and idx + 1 < len(normalized_args):
            skipped.add(normalized_args[idx + 1].lower())
            idx += 2
            continue
        idx += 1
    return skipped


def _build_pyright_targets(skipped_packages: set[str]) -> list[str]:
    """Determine which paths Pyright should type check for this run."""
    targets = list(PYRIGHT_BASE_TARGETS)
    for package, path in PYRIGHT_PACKAGE_TARGETS.items():
        if package in skipped_packages:
            continue
        targets.append(path)
    return targets


def _run_pyright(*, verbose: bool = False, skipped_packages: set[str] | None = None) -> bool:
    """Run Pyright across the repo."""
    _print_header("Python Type Checking (Pyright)")
    targets = _build_pyright_targets(skipped_packages or set())
    if not targets:
        info("No Pyright targets configured. Skipping.")
        return True
    cmd = ["uv", "run", "pyright", *targets]
    return _run_command(cmd, "pyright type checking", verbose=verbose)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _is_pyright_enforced() -> bool:
    """Determine if Pyright failures should fail the python stage."""
    return _is_truthy(os.environ.get(PYRIGHT_ENFORCE_ENV))


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

    cmd = ["uv", "run", "metta", "lint"]
    passed = _run_command(cmd, "Linting", verbose=verbose)
    return CheckResult("Lint", passed)


def _run_python_tests(
    *,
    verbose: bool = False,
    extra_args: Sequence[str] | None = None,
) -> CheckResult:
    """Run Pyright followed by Python tests and benchmarks."""
    normalized_args = _normalize_python_stage_args(extra_args)
    skipped_packages = _extract_skip_packages(normalized_args)

    enforce_pyright = _is_pyright_enforced()
    pyright_passed = _run_pyright(verbose=verbose, skipped_packages=skipped_packages)
    if not pyright_passed and not enforce_pyright:
        info(f"Pyright failed but {PYRIGHT_ENFORCE_ENV} is not set; continuing.")

    _print_header("Python Tests and Benchmarks")

    cmd = ["uv", "run", "metta", "pytest", "--ci", "--test", "--benchmark"]
    cmd.extend(normalized_args)
    tests_passed = _run_command(cmd, "Python tests and benchmarks", verbose=verbose)

    passed = tests_passed and (pyright_passed or not enforce_pyright)
    return CheckResult("Python Tests + Pyright", passed)


def _run_nim_tests(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Run Nim tests."""
    _ensure_no_extra_args("nim-tests", extra_args)
    _print_header("Nim Tests")

    cmd = ["uv", "run", "metta", "nimtest"]
    passed = _run_command(cmd, "Nim tests", verbose=verbose)
    return CheckResult("Nim Tests", passed)


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


def _setup_recipe_logging(log_file: Path, group: str) -> None:
    """Configure logging to write to file for recipe tests.

    All log messages (including from background threads) will be written to the log file.
    This keeps console output clean while still capturing detailed logs.
    Uses rotating file handler to prevent unbounded log growth.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler: max 10MB per file, keep 5 backups (50MB total)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,  # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    # Configure root logger: remove all handlers and add only file handler
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]  # Replace all handlers (removes console output)

    # Set metta logger to DEBUG (captures all metta.* logs in detail)
    # Other loggers will use their default levels (typically WARNING)
    metta_logger = logging.getLogger("metta")
    metta_logger.setLevel(logging.DEBUG)

    # Log run delimiter for easy identification in continuous stream
    separator = "=" * 80
    db_filename = f"{group}.sqlite"
    metta_logger.info(separator)
    metta_logger.info(f"CI RUN STARTED: {group}")
    metta_logger.info(f"Database: {db_filename}")
    metta_logger.info(f"Timestamp: {datetime.now().isoformat()}")
    metta_logger.info(separator)


def _run_cleanup_cancelled_runs(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Clean up cancelled workflow runs from concurrency settings."""
    _ensure_no_extra_args("cleanup-cancelled-runs", extra_args)
    _print_header("Cleanup Cancelled Runs")

    cmd = [
        "uv",
        "run",
        str(get_repo_root() / ".github/actions/cleanup-cancelled-runs/cleanup_cancelled_runs.py"),
    ]
    passed = _run_command(cmd, "Cleanup cancelled runs", verbose=verbose)
    return CheckResult("Cleanup Cancelled Runs", passed)


def _run_recipe_tests(
    *, verbose: bool = False, name_filter: str | None = None, no_interactive: bool = False, max_local_jobs: int = 2
) -> CheckResult:
    """Run recipe CI tests from stable recipes."""
    _print_header("Recipe CI Tests")

    try:
        # Get recipe CI jobs and group name
        all_jobs, group = get_ci_jobs()

        # Apply name filtering if provided
        if name_filter:
            recipe_jobs = [job for job in all_jobs if name_filter in job.name]
            if not recipe_jobs:
                error(f"No jobs matching '{name_filter}'")
                info(f"Available jobs: {', '.join(job.name for job in all_jobs)}")
                return CheckResult("Recipe Tests", False)
            info(f"Running {len(recipe_jobs)} job(s) matching '{name_filter}' (group: {group}):")
        else:
            recipe_jobs = all_jobs
            info(f"Running {len(recipe_jobs)} recipe CI tests (group: {group}):")

        if not recipe_jobs:
            info("No recipe CI tests found")
            return CheckResult("Recipe Tests", True)

        for job in recipe_jobs:
            console.print(f"  • {job.name}")

        # Use persistent directory for job state (already in .gitignore)
        jobs_dir = Path("train_dir/jobs")
        jobs_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging to file BEFORE creating JobManager
        log_file = jobs_dir / "ci_runner.log"
        _setup_recipe_logging(log_file, group)
        console.print(f"💡 Detailed logs: tail -f {log_file}\n")

        # Create JobManager after logging is configured
        manager = JobManager(base_dir=jobs_dir, max_local_jobs=max_local_jobs)

        # Submit, monitor, and report with group name
        all_passed = submit_monitor_and_report(
            manager,
            recipe_jobs,
            title="Recipe CI Tests",
            group=group,
            no_interactive=no_interactive,
        )

        if all_passed:
            success(f"✅ All {len(recipe_jobs)} recipe tests passed")
        else:
            error("❌ Some recipe tests failed - see details above")

        return CheckResult("Recipe Tests", all_passed)

    except Exception as e:
        error(f"Failed to run recipe tests: {e}")
        if verbose:
            console.print(traceback.format_exc())
        return CheckResult("Recipe Tests", False)


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


StageRunner = Callable[[bool, Sequence[str] | None, str | None, bool], CheckResult]

stages: dict[str, StageRunner] = {
    "lint": lambda v, args, name, _: _run_lint(verbose=v, extra_args=args),
    "python-tests-and-benchmarks": lambda v, args, name, _: _run_python_tests(verbose=v, extra_args=args),
    "cpp-tests": lambda v, args, name, _: _run_cpp_tests(verbose=v, extra_args=args),
    "cpp-benchmarks": lambda v, args, name, _: _run_cpp_benchmarks(verbose=v, extra_args=args),
    "nim-tests": lambda v, args, name, _: _run_nim_tests(verbose=v, extra_args=args),
    "recipe-tests": lambda v, args, name, ni: _run_recipe_tests(verbose=v, name_filter=name, no_interactive=ni),
    "cleanup-cancelled-runs": lambda v, args, name, _: _run_cleanup_cancelled_runs(verbose=v, extra_args=args),
}

# Stages that run by default when `metta ci` is called without --stage
# Excludes stages that require GitHub Actions context (e.g., cleanup-cancelled-runs)
DEFAULT_STAGES = {
    "lint",
    "python-tests-and-benchmarks",
    "cpp-tests",
    "cpp-benchmarks",
    "nim-tests",
    "recipe-tests",
}


def cmd_ci(
    ctx: typer.Context,
    stage: Annotated[
        str | None,
        typer.Option(help=f"Run specific stage: {', '.join(stages.keys())}"),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="Filter recipe-tests by job name substring"),
    ] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error", help="Don't stop on first failure")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output")] = False,
    no_interactive: Annotated[
        bool, typer.Option("--no-interactive", help="Disable live display for CI environments")
    ] = False,
):
    """Run CI checks locally to match remote CI behavior."""
    extra_args = list(getattr(ctx, "args", []))

    if extra_args and stage is None:
        error("Extra arguments require specifying a --stage.")
        raise typer.Exit(1)

    if name and stage != "recipe-tests":
        error("--name can only be used with --stage recipe-tests")
        raise typer.Exit(1)

    # If specific stage requested, run only that stage
    if stage:
        if stage not in stages:
            error(f"Unknown stage: {stage}")
            info(f"Valid stages: {', '.join(stages.keys())}")
            raise typer.Exit(1)

        # Run the specific stage
        result = stages[stage](verbose, extra_args, name, no_interactive)
        if result.passed:
            success(f"Stage '{stage}' passed!")
            sys.exit(0)
        else:
            error(f"Stage '{stage}' failed.")
            sys.exit(1)

    # Otherwise run all default stages (local development workflow)
    console.print(Panel.fit("[bold]Running All CI Checks[/bold]", border_style="cyan"))

    results: list[CheckResult] = []

    # Run only default stages (excludes stages that require GitHub Actions context)
    for stage_name, stage_func in stages.items():
        if stage_name not in DEFAULT_STAGES:
            continue
        result = stage_func(verbose, None, None, no_interactive)
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
