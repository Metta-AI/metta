from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from rich.markup import escape

from metta.setup.utils import error, info, step, success, warning

if TYPE_CHECKING:
    from metta.setup.tools.test_runner.test_python import PackageResult


@dataclass(slots=True)
class TestFailure:
    nodeid: str
    base_nodeid: str
    param_id: str | None
    message: str


@dataclass(slots=True)
class TestDuration:
    package_name: str
    nodeid: str
    duration: float


@dataclass(slots=True)
class PackageSummary:
    package_name: str
    target: str
    returncode: int
    duration: float
    total: int | None
    failures: list[TestFailure]
    durations: list[TestDuration]


def _load_report(report: Path) -> Mapping[str, Any] | None:
    if not report.exists():
        warning(f"Pytest JSON report missing: {report}")
        return None
    try:
        with report.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warning(f"Unable to read pytest report {report}: {exc}")
        return None
    if isinstance(data, Mapping):
        return data
    warning(f"Unexpected JSON structure in {report}")
    return None


def _summary_total(summary: Mapping[str, Any]) -> int | None:
    total = summary.get("total")
    if isinstance(total, int):
        return total
    values = [value for value in summary.values() if isinstance(value, (int, float))]
    if not values:
        return None
    return int(sum(values))


def _failure_message(entry: Mapping[str, Any]) -> str:
    for key in ("longreprtext", "longrepr"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for phase in ("setup", "call", "teardown"):
        phase_data = entry.get(phase)
        if isinstance(phase_data, Mapping):
            message = phase_data.get("longreprtext")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return ""


def _split_nodeid(nodeid: str) -> tuple[str, str | None]:
    if "[" not in nodeid:
        return nodeid, None
    base, _, remainder = nodeid.partition("[")
    return base, remainder.rstrip("]")


def _format_rerun_command(targets: Sequence[str]) -> str:
    if not targets:
        return "metta pytest"
    if len(targets) == 1:
        return f"metta pytest {targets[0]}"
    lines = ["metta pytest \\"]
    for target in targets[:-1]:
        lines.append(f"  {target} \\")
    lines.append(f"  {targets[-1]}")
    return "\n".join(lines)


def _normalize_nodeid(nodeid: str, target: str) -> str:
    path_part, sep, remainder = nodeid.partition("::")
    if not path_part:
        return nodeid
    target_path = Path(target)
    normalized_path = path_part
    target_prefix = target_path.as_posix()
    if not path_part.startswith(target_prefix):
        parent = target_path.parent.as_posix()
        if parent not in {"", "."}:
            normalized_path = f"{parent}/{path_part}"
    suffix = f"{sep}{remainder}" if sep else ""
    return f"{normalized_path}{suffix}"


def summarize_test_results(results: Sequence["PackageResult"]) -> list[PackageSummary]:
    summaries: list[PackageSummary] = []
    for result in results:
        data = _load_report(result.report_file)
        summary_section: Mapping[str, Any] = data.get("summary", {}) if data else {}
        total = _summary_total(summary_section) if summary_section else None

        failures: list[TestFailure] = []
        durations: list[TestDuration] = []
        tests = data.get("tests") if data else None
        if isinstance(tests, list):
            for entry in tests:
                if not isinstance(entry, Mapping):
                    continue
                nodeid = entry.get("nodeid")
                if not isinstance(nodeid, str):
                    continue
                normalized_nodeid = _normalize_nodeid(nodeid, result.target)
                duration_value = entry.get("duration")
                if isinstance(duration_value, (int, float)):
                    durations.append(
                        TestDuration(
                            package_name=result.package.name,
                            nodeid=normalized_nodeid,
                            duration=float(duration_value),
                        )
                    )
                if entry.get("outcome") not in {"failed", "error"}:
                    continue
                message = _failure_message(entry)
                base_nodeid, param_id = _split_nodeid(normalized_nodeid)
                failures.append(
                    TestFailure(
                        nodeid=normalized_nodeid,
                        base_nodeid=base_nodeid,
                        param_id=param_id,
                        message=message,
                    )
                )

        summaries.append(
            PackageSummary(
                package_name=result.package.name,
                target=result.target,
                returncode=result.returncode,
                duration=result.duration,
                total=total,
                failures=failures,
                durations=durations,
            )
        )
    return summaries


def log_results(summaries: Sequence[PackageSummary]) -> None:
    if not summaries:
        info("No Python test packages were selected.")
        return

    for summary in summaries:
        total_text = f"{summary.total} tests" if summary.total else "no tests"
        duration_text = f"{summary.duration:.1f}s"
        failure_count = len(summary.failures)

        if summary.returncode == 0 and failure_count == 0:
            success(f"✓ {summary.package_name} · {total_text} · {duration_text}")
        elif failure_count:
            error(f"✗ {summary.package_name} · {total_text} · {duration_text} · {failure_count} failing")
        else:
            error(f"✗ {summary.package_name} · {total_text} · {duration_text} · exit {summary.returncode}")


def report_failures(summaries: Sequence[PackageSummary]) -> dict[str, list[TestFailure]]:
    failure_map: dict[str, list[TestFailure]] = {
        summary.package_name: summary.failures for summary in summaries if summary.failures
    }

    if not summaries:
        return failure_map
    if not failure_map:
        if all(summary.returncode == 0 for summary in summaries):
            success("All Python test packages passed.")
        else:
            warning("Python test packages completed without JSON failures but returned non-zero exit codes.")
        return failure_map

    error("Failing tests:")
    for package_name, failures in failure_map.items():
        error(package_name, indent=2)
        grouped: dict[str, list[TestFailure]] = {}
        for failure in failures:
            grouped.setdefault(failure.base_nodeid, []).append(failure)
        for base_nodeid, grouped_failures in grouped.items():
            step(f"• {base_nodeid}", indent=4)
            for failure in grouped_failures:
                message_lines = [line for line in failure.message.strip().splitlines() if line][:40]
                header = escape(f"[{failure.param_id}]") if failure.param_id else ""
                escaped_lines = [escape(line) for line in message_lines]
                if header and escaped_lines:
                    info(f"{header} {escaped_lines[0]}", indent=6)
                    for line in escaped_lines[1:]:
                        info(line, indent=8)
                elif header:
                    info(header, indent=6)
                elif escaped_lines:
                    info(escaped_lines[0], indent=6)
                    for line in escaped_lines[1:]:
                        info(line, indent=8)
        unique_targets = list(dict.fromkeys(failure.nodeid for failure in failures))
        command_text = _format_rerun_command(unique_targets)
        info("Re-run locally:", indent=4)
        for line in command_text.splitlines():
            info(escape(line), indent=6)
    return failure_map


def write_github_summary(summaries: Sequence[PackageSummary]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines: list[str] = []
    lines.append("### Python test results")
    lines.append("")
    lines.append("| Package | Status | Rerun |")
    lines.append("| --- | --- | --- |")

    for summary in summaries:
        failures = summary.failures
        if failures:
            status = f"❌ {len(failures)} failing"
            targets = list(dict.fromkeys(failure.nodeid for failure in failures))
            command_text = _format_rerun_command(targets)
            if "\n" in command_text:
                rerun = f"<pre><code>{html.escape(command_text)}</code></pre>"
            else:
                rerun = f"`{command_text}`"
        elif summary.returncode != 0:
            status = f"⚠️ exit {summary.returncode}"
            rerun = f"`metta pytest {summary.target}`"
        else:
            total = summary.total or 0
            status = f"✅ {total} passed" if total else "✅ no tests"
            rerun = "—"
        lines.append(f"| {summary.package_name} | {status} | {rerun} |")

    lines.append("")

    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")

    durations: list[TestDuration] = []
    for summary in summaries:
        durations.extend(summary.durations)

    if not durations:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        header_lines = [
            "### Slowest tests",
            "",
            "| Duration (s) | Package | Test |",
            "| --- | --- | --- |",
        ]
        handle.write("\n".join(header_lines))
        for entry in durations:
            handle.write(f"\n| {entry.duration:.2f} | {entry.package_name} | `{entry.nodeid}` |")
        handle.write("\n\n")
