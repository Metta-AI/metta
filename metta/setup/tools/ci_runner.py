"""CI runner for local testing that matches remote CI behavior.

This tool is the single source of truth for CI checks.
Both local development (metta ci) and GitHub Actions call this same tool.

GitHub Actions workflow calls individual stages:
  - uv run metta ci --stage lint
  - uv run metta ci --stage pyright
  - uv run metta ci --stage python-tests-and-benchmarks
  - uv run metta ci --stage cpp-tests
  - uv run metta ci --stage cpp-benchmarks
  - uv run metta ci --stage recipe-tests

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

ALLOWED_SKIP_PACKAGES = {package.name.lower() for package in PYTEST_PACKAGES}


class CheckResult:
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
    except subprocess.CalledProcessError as exc:
        error(f"{description} failed")
        if not verbose:
            if exc.stdout:
                console.print(exc.stdout, markup=False)
            if exc.stderr:
                console.print(exc.stderr, markup=False)
        return False


def _run_lint(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
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
    _print_header("Python Tests and Benchmarks")

    cmd = ["uv", "run", "metta", "pytest", "--ci", "--test", "--benchmark"]
    cmd.extend(_normalize_python_stage_args(extra_args))
    passed = _run_command(cmd, "Python tests and benchmarks", verbose=verbose)

    return CheckResult("Python Tests", passed)


def _run_nim_tests(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    _ensure_no_extra_args("nim-tests", extra_args)
    _print_header("Nim Tests")

    cmd = ["uv", "run", "metta", "nimtest"]
    passed = _run_command(cmd, "Nim tests", verbose=verbose)
    return CheckResult("Nim Tests", passed)


def _run_cpp_tests(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    _ensure_no_extra_args("cpp-tests", extra_args)
    _print_header("C++ Tests")

    cmd = ["uv", "run", "metta", "cpptest", "--test"]
    if verbose:
        cmd.append("--verbose")
    passed = _run_command(cmd, "C++ unit tests", verbose=verbose)

    return CheckResult("C++ Tests", passed)


def _run_cpp_benchmarks(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    _ensure_no_extra_args("cpp-benchmarks", extra_args)
    _print_header("C++ Benchmarks")

    cmd = ["uv", "run", "metta", "cpptest", "--benchmark"]
    if verbose:
        cmd.append("--verbose")
    passed = _run_command(cmd, "C++ benchmarks", verbose=verbose)

    return CheckResult("C++ Benchmarks", passed)


def _run_cleanup_cancelled_runs(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    _ensure_no_extra_args("cleanup-cancelled-runs", extra_args)
    _print_header("Cleanup Cancelled Runs")

    cmd = [
        "uv",
        "run",
        str(get_repo_root() / ".github/actions/cleanup-cancelled-runs/cleanup_cancelled_runs.py"),
    ]
    passed = _run_command(cmd, "Cleanup cancelled runs", verbose=verbose)
    return CheckResult("Cleanup Cancelled Runs", passed)


def _run_recipe_tests(*, verbose: bool = False, name_filter: str | None = None, **_kwargs) -> CheckResult:
    _print_header("Recipe CI Tests")

    cmd = ["uv", "run", "./devops/stable/cli.py", "--suite=ci", "--skip-submitting-metrics"]
    if name_filter:
        cmd.extend(["--job", name_filter])

    passed = _run_command(cmd, "Recipe CI tests", verbose=verbose)
    return CheckResult("Recipe Tests", passed)


_CHECK_PYRIGHT_PACKAGES = [
    "packages/cogames",
    "app_backend",
]


def _run_pyright(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    _ensure_no_extra_args("pyright", extra_args)
    _print_header("Pyright")

    cmd = ["uv", "run", "pyright", *_CHECK_PYRIGHT_PACKAGES]
    passed = _run_command(cmd, "Pyright", verbose=verbose)
    return CheckResult("Pyright", passed)


def _run_proto_check(*, verbose: bool = False, extra_args: Sequence[str] | None = None) -> CheckResult:
    """Check that generated protobuf files are up to date with .proto sources.

    Limitation: This check won't detect orphaned _pb2.py files when a .proto
    file is deleted. It only verifies that existing .proto files produce the
    same output as the committed generated files.
    """
    _ensure_no_extra_args("proto-check", extra_args)
    _print_header("Proto Staleness Check")

    repo_root = get_repo_root()
    generated_globs = ["*_pb2.py", "*_pb2.pyi"]

    # Regenerate protos
    gen_cmd = [sys.executable, str(repo_root / "scripts" / "generate_protos.py")]
    info(f"Running: {_format_cmd_for_display(gen_cmd)}")
    gen_result = subprocess.run(gen_cmd, cwd=repo_root, capture_output=True, text=True)
    if gen_result.returncode != 0:
        error("Proto generation failed")
        if gen_result.stdout:
            console.print(gen_result.stdout, markup=False)
        if gen_result.stderr:
            console.print(gen_result.stderr, markup=False)
        return CheckResult("Proto Check", False)

    # Check for uncommitted changes to tracked generated files
    diff_cmd = ["git", "diff", "--exit-code", "--", *generated_globs]
    info(f"Running: {_format_cmd_for_display(diff_cmd)}")
    diff_result = subprocess.run(diff_cmd, cwd=repo_root, capture_output=True, text=True)
    if diff_result.returncode != 0:
        error("Generated proto files are out of date. Run: python scripts/generate_protos.py")
        if verbose or diff_result.stdout:
            console.print(diff_result.stdout, markup=False)
        return CheckResult("Proto Check", False)

    # Check for untracked generated files (new .proto files without committed outputs)
    untracked_cmd = ["git", "ls-files", "--others", "--exclude-standard", "--", *generated_globs]
    info(f"Running: {_format_cmd_for_display(untracked_cmd)}")
    untracked_result = subprocess.run(untracked_cmd, cwd=repo_root, capture_output=True, text=True)
    untracked_files = untracked_result.stdout.strip()
    if untracked_files:
        error("Untracked generated proto files. Run: python scripts/generate_protos.py && git add <files>")
        console.print(untracked_files, markup=False)
        return CheckResult("Proto Check", False)

    success("Proto files are up to date")
    return CheckResult("Proto Check", True)


def _print_summary(results: list[CheckResult]) -> None:
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
    "pyright": lambda v, args, name, _: _run_pyright(verbose=v, extra_args=args),
    "proto-check": lambda v, args, name, _: _run_proto_check(verbose=v, extra_args=args),
    "python-tests-and-benchmarks": lambda v, args, name, _: _run_python_tests(verbose=v, extra_args=args),
    "cpp-tests": lambda v, args, name, _: _run_cpp_tests(verbose=v, extra_args=args),
    "cpp-benchmarks": lambda v, args, name, _: _run_cpp_benchmarks(verbose=v, extra_args=args),
    "nim-tests": lambda v, args, name, _: _run_nim_tests(verbose=v, extra_args=args),
    "recipe-tests": lambda v, args, name, ni: _run_recipe_tests(verbose=v, name_filter=name),
    "cleanup-cancelled-runs": lambda v, args, name, _: _run_cleanup_cancelled_runs(verbose=v, extra_args=args),
}

DEFAULT_STAGES = {
    "lint",
    "proto-check",
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

    if stage:
        if stage not in stages:
            error(f"Unknown stage: {stage}")
            info(f"Valid stages: {', '.join(stages.keys())}")
            raise typer.Exit(1)

        result = stages[stage](verbose, extra_args, name, no_interactive)
        if result.passed:
            success(f"Stage '{stage}' passed!")
            sys.exit(0)
        else:
            error(f"Stage '{stage}' failed.")
            sys.exit(1)

    console.print(Panel.fit("[bold]Running All CI Checks[/bold]", border_style="cyan"))

    results: list[CheckResult] = []

    for stage_name, stage_func in stages.items():
        if stage_name not in DEFAULT_STAGES:
            continue
        result = stage_func(verbose, None, None, no_interactive)
        results.append(result)
        if not result.passed and not continue_on_error:
            _print_summary(results)
            error(f"Stage '{stage_name}' failed. Fix errors and try again.")
            raise typer.Exit(1)

    _print_summary(results)

    all_passed = all(r.passed for r in results)
    if all_passed:
        success("All CI checks passed!")
        sys.exit(0)
    else:
        error("Some CI checks failed.")
        sys.exit(1)
