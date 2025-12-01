#!/usr/bin/env python3
"""Run a `metta ci` stage and publish a concise GitHub summary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

FAIL_MARKER = "Re-run locally:"


def _title_for_stage(stage: str) -> str:
    return stage.replace("-", " ").title()


def _write_summary(title: str, lines: Iterable[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write(f"## {title}\n")
        for line in lines:
            handle.write(f"{line}\n")
        handle.write("\n")


def _extract_python_failures(output: str) -> List[str]:
    failures: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("FAILED "):
            failures.append(line[7:].strip())
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for item in failures:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _extract_cpp_failures(output: str) -> List[str]:
    failures: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("[  FAILED  ]"):
            failures.append(line[12:].strip())
        elif line.startswith("//") and " FAILED" in line:
            failures.append(line.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for item in failures:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _extract_re_run_commands(output: str) -> List[str]:
    commands: list[str] = []
    lines = output.splitlines()
    for idx, raw in enumerate(lines):
        if raw.strip().startswith(FAIL_MARKER):
            if idx + 1 < len(lines):
                cmd = lines[idx + 1].strip()
                if cmd:
                    commands.append(cmd)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in commands:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _summarize(stage: str, success: bool, output: str) -> List[str]:
    bullet = "✅" if success else "❌"
    lines: list[str] = [f"- {bullet} Stage `{stage}` {'passed' if success else 'failed'}."]
    if success:
        return lines

    if stage == "python-tests-and-benchmarks":
        failures = _extract_python_failures(output)
        commands = _extract_re_run_commands(output)
        if failures:
            lines.append("- Failing tests:")
            for entry in failures:
                entry_norm = entry.strip()
                if not entry_norm:
                    continue
                if " - " in entry_norm:
                    test, detail = entry_norm.split(" - ", 1)
                    lines.append(f"  - `{test.strip()}` — {detail.strip()}")
                else:
                    lines.append(f"  - `{entry_norm}`")
        else:
            lines.append("- Failed tests: see log output above.")

        if commands:
            if len(commands) == 1:
                lines.append(f"- Re-run locally: `{commands[0]}`")
            else:
                lines.append("- Re-run locally:")
                for command in commands:
                    lines.append(f"  - `{command}`")
    elif stage == "cpp-tests":
        failures = _extract_cpp_failures(output)
        if failures:
            lines.append("- Failing C++ tests:")
            for entry in failures:
                lines.append(f"  - `{entry}`")
        else:
            lines.append("- Failing C++ tests: see log output above.")
        lines.append("- Re-run locally: `metta cpptest --test`")
    elif stage == "cpp-benchmarks":
        lines.append("- Re-run locally: `metta cpptest --benchmark`")
        lines.append("- See log output above for benchmark failures.")
    else:
        lines.append("- See log output above for details.")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a metta CI stage")
    parser.add_argument("stage", help="Stage name to pass to `metta ci`.")
    args, remainder = parser.parse_known_args()

    cmd = ["uv", "run", "metta", "ci", "--stage", args.stage]
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    cmd.extend(remainder)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    combined_output = proc.stdout + "\n" + proc.stderr
    summary_lines = _summarize(args.stage, proc.returncode == 0, combined_output)
    _write_summary(_title_for_stage(args.stage), summary_lines)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
