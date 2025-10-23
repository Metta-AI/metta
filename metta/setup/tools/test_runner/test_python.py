import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Mapping, Sequence

import typer
from pydantic import BaseModel

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, step, success, warning


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
    Package(name="mettascope", target=Path("mettascope/tests")),
    Package(name="agent", target=Path("agent/tests")),
    Package(name="app_backend", target=Path("app_backend/tests")),
    Package(name="common", target=Path("common/tests")),
    Package(name="codebot", target=Path("packages/codebot/tests")),
    Package(name="cogames", target=Path("packages/cogames/tests")),
    Package(name="gitta", target=Path("packages/gitta/tests")),
    Package(name="mettagrid", target=Path("packages/mettagrid/tests")),
    Package(name="cortex", target=Path("packages/cortex/tests")),
)


DEFAULT_FLAGS: tuple[str, ...] = ("--benchmark-disable", "-n", "auto")
CI_FLAGS: tuple[str, ...] = (
    "-n",
    "4",
    "--timeout=100",
    "--timeout-method=thread",
    "--benchmark-skip",
    "--disable-warnings",
    "--durations=10",
    "--color=yes",
)


def _run_command(args: Sequence[str]) -> int:
    info(f"→ {' '.join(args)}")
    completed = subprocess.run(args, cwd=get_repo_root(), check=False)
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
    output: str
    report_file: Path
    duration: float


@dataclass(slots=True)
class TestFailure:
    nodeid: str
    message: str


@dataclass(slots=True)
class PackageSummary:
    result: PackageResult
    summary: dict[str, int]
    failures: list[TestFailure]


def _total_tests(summary: Mapping[str, Any]) -> int:
    total = summary.get("total")
    if isinstance(total, int):
        return total
    values = [value for value in summary.values() if isinstance(value, (int, float))]
    return int(sum(values))


def _run_ci_package(
    package: Package,
    target: str,
    base_cmd: Sequence[str],
    report_dir: Path,
) -> PackageResult:
    report_file = report_dir / f"{package.key}.json"
    if report_file.exists():
        report_file.unlink()
    args = [*base_cmd, target, "--json-report", f"--json-report-file={report_file}"]
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=get_repo_root(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration = time.perf_counter() - start
    output = completed.stdout or ""
    return PackageResult(
        package=package,
        target=target,
        returncode=completed.returncode,
        output=output,
        report_file=report_file,
        duration=duration,
    )


def _execute_ci_packages(
    packages: Sequence[Package],
    targets: Sequence[str],
    base_cmd: Sequence[str],
    report_dir: Path,
) -> list[PackageResult]:
    index_map = {package.key: index for index, package in enumerate(packages)}
    futures = []
    results: list[PackageResult] = []
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        for package, target in zip(packages, targets, strict=True):
            futures.append(pool.submit(_run_ci_package, package, target, base_cmd, report_dir))

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: index_map.get(item.package.key, len(index_map)))
    return results


def _load_json_report(report: Path) -> dict[str, Any] | None:
    if not report.exists():
        warning(f"Pytest JSON report missing: {report}")
        return None
    try:
        with report.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        warning(f"Unable to parse pytest JSON report {report}: {exc}")
        return None
    if not isinstance(data, dict):
        warning(f"Unexpected JSON structure for pytest report {report}")
        return None
    return data


def _format_longrepr(longrepr: Any) -> str:
    if isinstance(longrepr, str):
        return longrepr
    if isinstance(longrepr, list):
        lines = [str(line) for line in longrepr if line is not None]
        return "\n".join(lines)
    if isinstance(longrepr, Mapping):
        reprcrash = longrepr.get("reprcrash")
        if isinstance(reprcrash, Mapping):
            message = reprcrash.get("message")
            if isinstance(message, str) and message:
                return message
        reprentries = longrepr.get("reprentries")
        if isinstance(reprentries, list):
            formatted: list[str] = []
            for entry in reprentries:
                if isinstance(entry, Mapping):
                    lines = entry.get("lines")
                    if isinstance(lines, list):
                        formatted.extend(str(line) for line in lines if line is not None)
            if formatted:
                return "\n".join(formatted)
    return ""


def _collect_captured_sections(section: Mapping[str, Any]) -> list[str]:
    collected: list[str] = []
    sections = section.get("sections")
    if isinstance(sections, list):
        for entry in sections:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                heading, content = entry
                heading_text = str(heading).strip()
                content_text = str(content).strip()
                if content_text:
                    collected.append(f"{heading_text}:\n{content_text}")
    return collected


def _extract_failure_message(test: Mapping[str, Any]) -> str:
    for phase in ("setup", "call", "teardown"):
        section = test.get(phase)
        if not isinstance(section, Mapping):
            continue
        if section.get("outcome") != "failed":
            continue
        message = _format_longrepr(section.get("longrepr"))
        if not message:
            longrepr_text = section.get("longreprtext")
            if isinstance(longrepr_text, str):
                message = longrepr_text
        extra = _collect_captured_sections(section)
        parts = [part for part in [message, *extra] if part]
        if parts:
            return "\n\n".join(parts)
    # Fallback to test-level longrepr if phases did not match
    message = _format_longrepr(test.get("longrepr"))
    if message:
        return message
    outcome = test.get("outcome")
    nodeid = test.get("nodeid")
    return f"{nodeid} -> {outcome}"


def _truncate_message(message: str, max_lines: int = 80) -> str:
    lines = message.splitlines()
    if len(lines) <= max_lines:
        return message
    truncated = "\n".join(lines[:max_lines])
    return f"{truncated}\n… (truncated)"


def _indent_block(message: str, indent: int) -> str:
    padding = " " * indent
    return "\n".join(f"{padding}{line}" if line else "" for line in message.splitlines())


def _build_package_summaries(results: Sequence[PackageResult]) -> list[PackageSummary]:
    summaries: list[PackageSummary] = []
    for result in results:
        data = _load_json_report(result.report_file)
        summary_data: dict[str, int] = {}
        failures: list[TestFailure] = []
        if data:
            summary_section = data.get("summary", {})
            if isinstance(summary_section, Mapping):
                summary_data = {
                    key: int(value) for key, value in summary_section.items() if isinstance(value, (int, float))
                }

            tests = data.get("tests", [])
            if isinstance(tests, list):
                for entry in tests:
                    if not isinstance(entry, Mapping):
                        continue
                    outcome = entry.get("outcome")
                    if outcome not in {"failed", "error"}:
                        continue
                    nodeid = entry.get("nodeid")
                    if not isinstance(nodeid, str):
                        continue
                    message = _truncate_message(_extract_failure_message(entry))
                    failures.append(TestFailure(nodeid=nodeid, message=message))

        summaries.append(PackageSummary(result=result, summary=summary_data, failures=failures))
    return summaries


def _log_package_results(summaries: Sequence[PackageSummary]) -> None:
    if not summaries:
        info("No Python test packages were selected.")
        return

    for package_summary in summaries:
        result = package_summary.result
        summary = package_summary.summary
        total = _total_tests(summary)
        total_text = f"{total} test{'s' if total != 1 else ''}" if total else "no tests"
        duration_text = f"{result.duration:.1f}s"
        failure_count = len(package_summary.failures)

        if result.returncode == 0 and failure_count == 0:
            success(f"✓ {result.package.name} · {total_text} · {duration_text}")
        elif failure_count > 0:
            error(f"✗ {result.package.name} · {total_text} · {duration_text} · {failure_count} failing")
        else:
            error(f"✗ {result.package.name} · {total_text} · {duration_text} · pytest exit {result.returncode}")


def _emit_failure_summary(summaries: Sequence[PackageSummary]) -> dict[str, list[TestFailure]]:
    failure_map: dict[str, list[TestFailure]] = {}
    for package_summary in summaries:
        if package_summary.failures:
            failure_map[package_summary.result.package.name] = package_summary.failures

    if not failure_map:
        if all(summary.result.returncode == 0 for summary in summaries):
            success("All Python test packages passed.")
        else:
            warning("Python test packages completed without JSON failures but returned non-zero exit codes.")
        return failure_map

    error("Failing tests:")
    for package_name, failures in failure_map.items():
        error(package_name, indent=2)
        rerun_targets = " ".join(failure.nodeid for failure in failures)
        for failure in failures:
            step(f"• {failure.nodeid}", indent=4)
            if failure.message:
                print(_indent_block(failure.message, 6))
        info(f"Re-run locally: uv run pytest {rerun_targets}", indent=4)
    return failure_map


def _write_github_summary(summaries: Sequence[PackageSummary], failure_map: Mapping[str, list[TestFailure]]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines: list[str] = []
    lines.append("### Python test results")
    lines.append("")
    lines.append("| Package | Status | Rerun |")
    lines.append("| --- | --- | --- |")

    for package_summary in summaries:
        result = package_summary.result
        failures = failure_map.get(result.package.name, [])
        total = _total_tests(package_summary.summary)
        status: str
        if failures:
            status = f"❌ {len(failures)} failing"
            rerun_targets = " ".join(failure.nodeid for failure in failures)
            rerun_command = f"`uv run pytest {rerun_targets}`"
        elif result.returncode != 0:
            status = f"⚠️ exit {result.returncode}"
            rerun_command = f"`uv run pytest {result.target}`"
        else:
            status = f"✅ {total} passed" if total else "✅ no tests"
            rerun_command = "—"
        lines.append(f"| {result.package.name} | {status} | {rerun_command} |")

    lines.append("")

    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


@app.callback()
def run(
    ctx: typer.Context,
    targets: Annotated[list[str] | None, typer.Argument(help="Explicit pytest targets.")] = None,
    ci: bool = typer.Option(False, "--ci", help="Use CI-style settings and parallel suite execution."),
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
        base_cmd = [*cmd, *CI_FLAGS, *extra_args]
        with tempfile.TemporaryDirectory(prefix="pytest-json-") as temp_dir:
            report_dir = Path(temp_dir)
            results = _execute_ci_packages(selected, resolved_targets, base_cmd, report_dir)
            summaries = _build_package_summaries(results)
            _log_package_results(summaries)
            failure_map = _emit_failure_summary(summaries)
            _write_github_summary(summaries, failure_map)
            exit_code = max((result.returncode for result in results), default=0)
        raise typer.Exit(exit_code)

    cmd.extend(DEFAULT_FLAGS)
    if changed:
        cmd.append("--testmon")
    cmd.extend(extra_args)
    cmd.extend(resolved_targets)
    raise typer.Exit(_run_command(cmd))
