#!/usr/bin/env python3

"""Summarize failing pytest cases from a JUnit XML report.

Usage:
    summarize_junit.py path/to/report.xml [path/to/output.md]

Prints a concise summary to stdout (including GitHub log annotations) and,
optionally, writes the same content to the provided markdown file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from metta.setup.tools.test_runner.junit_summary import emit_annotations, summarize_report


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        print("Usage: summarize_junit.py REPORT.xml [OUTPUT.md]", file=sys.stderr)
        raise SystemExit(2)

    report = Path(sys.argv[1])
    if not report.exists():
        raise SystemExit(f"Report not found: {report}")

    summary = summarize_report(report)
    emit_annotations(report)
    print(summary)

    if len(sys.argv) == 3:
        output = Path(sys.argv[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
