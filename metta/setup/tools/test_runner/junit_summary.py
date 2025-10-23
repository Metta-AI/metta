from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, NamedTuple


class Failure(NamedTuple):
    file: str | None
    line: int | None
    testcase: str
    message: str
    details: str
    target: str


def merge_reports(report_paths: Iterable[Path], output: Path) -> None:
    suites: list[ET.Element] = []
    for path in report_paths:
        if not path or not path.exists():
            continue
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag == "testsuite":
            suites.append(root)
        elif root.tag == "testsuites":
            suites.extend(root.findall("testsuite"))
        else:
            suites.append(root)

    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("testsuites")
    for suite in suites:
        root.append(suite)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def _iter_failures(xml_path: Path) -> list[Failure]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    failures: list[Failure] = []

    for testcase in root.iter("testcase"):
        failure_nodes = list(testcase.findall("failure")) + list(testcase.findall("error"))
        if not failure_nodes:
            continue

        file_attr = testcase.attrib.get("file")
        line_attr = testcase.attrib.get("line")
        try:
            line_val = int(line_attr) if line_attr is not None else None
        except ValueError:
            line_val = None

        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "<unknown>")
        testcase_name = f"{classname}.{name}" if classname else name
        target = file_attr or classname.replace(".", "/")
        pytest_target = f"{target}::{name}" if target else name

        for node in failure_nodes:
            message = node.attrib.get("message", "").strip()
            details = (node.text or "").strip()
            if not message and details:
                message = details.splitlines()[0]
            failures.append(
                Failure(
                    file=file_attr,
                    line=line_val,
                    testcase=testcase_name,
                    message=message,
                    details=details,
                    target=pytest_target,
                )
            )
    return failures


def summarize_report(xml_path: Path) -> str:
    if not xml_path.exists():
        return "⚠️ Pytest report not found; no summary generated."

    failures = _iter_failures(xml_path)
    if not failures:
        return "✅ Pytest reported no failures.\n"

    lines: list[str] = ["🔴 Pytest failures", ""]
    for failure in failures:
        header = f"- `{failure.testcase}`"
        if failure.message:
            header += f": {failure.message}"
        lines.append(header)
        lines.append(f"  - reproducible via `uv run metta pytest -- {failure.target}`")
        if failure.details:
            snippet = "\n".join(f"    {line}" for line in failure.details.splitlines()[:20])
            lines.append("  - details:\n" + snippet)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def emit_annotations(xml_path: Path) -> None:
    for failure in _iter_failures(xml_path):
        if failure.file:
            location = f"file={failure.file}"
            if failure.line:
                location += f",line={failure.line}"
            message = failure.message or failure.testcase
            print(f"::error {location}::{message}")
