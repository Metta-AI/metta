import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable, Sequence

import typer
from pydantic import BaseModel

from metta.common.util.fs import get_repo_root
from metta.setup.tools.test_runner.summary import (
    log_results,
    report_failures,
    summarize_test_results,
    write_github_summary,
    write_slow_tests_github_summary,
)
from metta.setup.utils import error, info


class Package(BaseModel):
    name: str
    target: Path

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def target_path(self) -> Path:
        root = get_repo_root()
        if self.target.is_absolute():
            return self.target
        return root / self.target


PACKAGES: tuple[Package, ...] = (
    Package(name="tests", target=Path("tests")),
    Package(name="agent", target=Path("agent/tests")),
    Package(name="app_backend", target=Path("app_backend/tests")),
    Package(name="common", target=Path("common/tests")),
    Package(name="codebot", target=Path("packages/codebot/tests")),
    Package(name="cogames", target=Path("packages/cogames/tests")),
    Package(name="gitta", target=Path("packages/gitta/tests")),
    Package(name="mettagrid", target=Path("packages/mettagrid/tests")),
    Package(name="cortex", target=Path("packages/cortex/tests")),
)


def _run_command(args: Sequence[str], env: dict[str, str] | None = None) -> int:
    info(f"→ {' '.join(args)}")
    completed = subprocess.run(args, cwd=get_repo_root(), check=False, env={**os.environ, **(env or {})})
    return completed.returncode


def _resolve_package_targets(
    include: Iterable[str],
    exclude: Iterable[str],
) -> list[Package]:
    package_map = {package.key: package for package in PACKAGES}
    include_keys = [name.lower() for name in include]
    exclude_keys = {name.lower() for name in exclude}

    selected: list[Package]
    if include_keys:
        missing = [name for name in include_keys if name not in package_map]
        if missing:
            raise ValueError(f"Unknown suite(s): {', '.join(missing)}")
        selected = [package_map[name] for name in include_keys]
    else:
        selected = list(PACKAGES)

    if exclude_keys:
        selected = [package for package in selected if package.key not in exclude_keys]

    return selected


def _collect_package_targets(packages: Iterable[Package]) -> list[str]:
    targets: list[Path] = []
    for package in packages:
        path = package.target_path
        if not path.exists():
            raise ValueError(f"Test package directory missing: {path}")
        targets.append(path)
    if not targets:
        raise ValueError("No package targets resolved.")
    return [str(t.relative_to(get_repo_root())) for t in targets]


app = typer.Typer(
    help="Python test runner",
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)


@dataclass(slots=True)
class PackageResult:
    package: Package
    target: str
    returncode: int
    report_file: Path
    duration: float


def _execute_ci_packages(
    packages: Sequence[Package],
    targets: Sequence[str],
    base_cmd: Sequence[str],
    report_dir: Path,
) -> list[PackageResult]:
    index_map = {package.key: index for index, package in enumerate(packages)}
    futures = []
    results: list[PackageResult] = []

    def _run_ci_package(
        package: Package,
        target: str,
        base_cmd: Sequence[str],
        report_dir: Path,
    ) -> PackageResult:
        report_file = report_dir / f"{package.key}.json"
        if report_file.exists():
            report_file.unlink()
        start = time.perf_counter()
        returncode = _run_command(
            [*base_cmd, target, "--json-report", f"--json-report-file={report_file}"],
            env={"DD_SERVICE": package.name},
        )
        duration = time.perf_counter() - start
        return PackageResult(
            package=package,
            target=target,
            returncode=returncode,
            report_file=report_file,
            duration=duration,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for package, target in zip(packages, targets, strict=True):
            futures.append(pool.submit(_run_ci_package, package, target, base_cmd, report_dir))

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: index_map.get(item.package.key, len(index_map)))
    return results


@app.callback()
def run(
    ctx: typer.Context,
    targets: Annotated[list[str] | None, typer.Argument(help="Explicit pytest targets.")] = None,
    ci: bool = typer.Option(False, "--ci", help="Use CI-style settings and parallel suite execution."),
    test: bool = typer.Option(False, "--test", help="Run unit tests (default if no flags specified)."),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run benchmarks."),
    packages: Annotated[
        list[str] | None, typer.Option("--package", "-p", help="Limit to specific named package(s).")
    ] = None,
    skip_packages: Annotated[
        list[str] | None, typer.Option("--skip-package", help="Exclude package(s) by name.")
    ] = None,
    changed: bool = typer.Option(
        False,
        "--changed",
        help="Run only tests impacted by staged/changed files using pytest-testmon.",
    ),
) -> None:
    extra_args = list(ctx.args or [])
    target_args = targets or []
    package_args = packages or []
    skip_package_args = skip_packages or []

    # Determine what to run based on flags
    # Default (no flags): tests only
    # --test only: tests only
    # --benchmark only: benchmarks only
    # --test --benchmark: both
    run_tests = test or not benchmark  # Default to tests if no flags
    run_benchmarks = benchmark

    cmd = ["uv", "run", "pytest"]
    if target_args:
        if package_args or skip_package_args or changed or ci:
            error("Explicit targets cannot be combined with suite filters, --changed, or --ci.")
            raise typer.Exit(1)
        exit_code = _run_command([*cmd, *target_args, *extra_args])
        raise typer.Exit(exit_code)

    try:
        selected = _resolve_package_targets(package_args, skip_package_args)
        resolved_targets = _collect_package_targets(selected)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc

    if ci:
        base_cmd = [
            *cmd,
            "-n",
            "4",
            "--timeout=240",
            "--timeout-method=thread",
            "--disable-warnings",
            "--color=no",
            "-q",
            "-o",
            "console_output_style=count",
        ]

        # Apply benchmark filtering for CI mode
        if run_benchmarks and not run_tests:
            base_cmd.append("--benchmark-only")
        elif run_tests and not run_benchmarks:
            base_cmd.append("--benchmark-skip")
        # else: both enabled, no filtering

        base_cmd.extend(extra_args)

        with tempfile.TemporaryDirectory(prefix="pytest-json-") as temp_dir:
            report_dir = Path(temp_dir)
            results = _execute_ci_packages(selected, resolved_targets, base_cmd, report_dir)
            summaries = summarize_test_results(results)
            log_results(summaries)
            report_failures(summaries)
            write_github_summary(summaries)
            write_slow_tests_github_summary(summaries)
            exit_code = max((result.returncode for result in results), default=0)
        raise typer.Exit(exit_code)

    cmd.extend(["-n", "auto"])

    # Apply benchmark filtering for non-CI mode
    if run_benchmarks and not run_tests:
        cmd.append("--benchmark-only")
    elif run_tests and not run_benchmarks:
        cmd.append("--benchmark-disable")
    # else: both enabled, no filtering

    if changed:
        cmd.append("--testmon")
    cmd.extend(extra_args)
    cmd.extend(resolved_targets)
    raise typer.Exit(_run_command(cmd))
