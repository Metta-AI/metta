#!/usr/bin/env python3

"""Summarize failing pytest cases from a JUnit XML report.

Usage:
    summarize_junit.py path/to/report.xml [path/to/output.md]

Prints a concise summary to stdout (including GitHub log annotations) and,
optionally, writes the same content to the provided markdown file.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, NamedTuple

SUMMARY_HEADER = "🔴 Pytest failures"
REPRO_TEMPLATE = "uv run metta pytest -- {target}"


class Failure(NamedTuple):
    file: str | None
    line: int | None
    testcase: str
    message: str
    details: str
    target: str


def _iter_failures(xml_path: Path) -> Iterable[Failure]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise SystemExit(f"Unable to parse JUnit report: {exc}") from exc

    root = tree.getroot()
    for testcase in root.iter("testcase"):
        failures = list(testcase.findall("failure")) + list(testcase.findall("error"))
        if not failures:
            continue

        file_attr = testcase.attrib.get("file")
        try:
            line_attr = int(testcase.attrib["line"])
        except (KeyError, ValueError):
            line_attr = None

        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "<unknown>")
        testcase_name = f"{classname}.{name}" if classname else name
        target = file_attr or classname.replace(".", "/")
        pytest_target = f"{target}::{name}" if target else name

        for failure in failures:
            message = failure.attrib.get("message", "").strip()
            details = (failure.text or "").strip()
            yield Failure(
                file=file_attr,
                line=line_attr,
                testcase=testcase_name,
                message=message or details.splitlines()[0] if details else "",
                details=details,
                target=pytest_target,
            )


def _format_failure(failure: Failure) -> str:
    header = f"- `{failure.testcase}`"
    if failure.message:
        header += f": {failure.message}"
    repro = REPRO_TEMPLATE.format(target=failure.target)
    lines = [header, f"  - reproducible via `{repro}`"]
    if failure.details:
        snippet = "\n".join(f"    {line}" for line in failure.details.splitlines()[:20])
        lines.append("  - details:\n" + snippet)
    return "\n".join(lines)


def _emit_annotations(failure: Failure) -> None:
    if failure.file:
        location = f"file={failure.file}"
        if failure.line:
            location += f",line={failure.line}"
        message = failure.message or failure.testcase
        print(f"::error {location}::{message}")


def summarize(xml_path: Path) -> str:
    failures = list(_iter_failures(xml_path))
    if not failures:
        return "✅ Pytest reported no failures."

    for failure in failures:
        _emit_annotations(failure)

    parts = [SUMMARY_HEADER, ""]
    for failure in failures:
        parts.append(_format_failure(failure))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        print("Usage: summarize_junit.py REPORT.xml [OUTPUT.md]", file=sys.stderr)
        raise SystemExit(2)

    report = Path(sys.argv[1])
    if not report.exists():
        raise SystemExit(f"Report not found: {report}")

    summary = summarize(report)
    print(summary)

    if len(sys.argv) == 3:
        output = Path(sys.argv[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
