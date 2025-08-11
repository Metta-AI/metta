#!/usr/bin/env python3

"""
Shared utilities for mettagrid build scripts.
"""

import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CompilerMessage:
    """Represents a compiler warning or error."""

    file_path: str
    line_number: Optional[int]
    severity: str  # 'warning', 'error', 'note'
    message: str
    flag: Optional[str]  # e.g., '-Wconversion'
    raw_line: str  # Store the original line for context

    @property
    def location(self) -> str:
        """Get formatted location string."""
        loc = self.file_path
        if self.line_number:
            loc += f":{self.line_number}"
        return loc


class BuildChecker:
    """Analyzes build output for warnings and errors."""

    # GCC/Clang pattern (Linux compilers)
    GCC_CLANG_PATTERN = re.compile(
        r"^(?P<file>[^:]+):(?P<line>\d+)(?::\d+)?:\s*"
        r"(?P<severity>warning|error|note):\s*(?P<message>.*?)(?:\s*\[(?P<flag>-W[^\]]+)\])?$"
    )

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        # The repo root is the parent of mettagrid/
        self.repo_root = self.project_root.parent.resolve()
        self.messages: list[CompilerMessage] = []
        self.build_failed = False
        self.runtime_issues = []  # Simple list for runtime problems

    def parse_build_output(self, output: str) -> None:
        """Parse build output and extract warnings/errors."""
        parsed_count = 0
        total_lines = 0

        for line in output.splitlines():
            total_lines += 1
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for runtime issues (simple patterns)
            if any(
                pattern in line_stripped
                for pattern in [
                    "AddressSanitizer: SEGV",
                    "AddressSanitizer:DEADLYSIGNAL",
                    "SUMMARY: AddressSanitizer:",
                    "==ABORTING",
                    "Segmentation fault",
                    "Assertion failed",
                    "Errors while running CTest",
                    "Aborted",
                    "core dumped",
                ]
            ):
                self.runtime_issues.append(line_stripped)
                self.build_failed = True

            match = self.GCC_CLANG_PATTERN.match(line_stripped)
            if match:
                parsed_count += 1
                groups = match.groupdict()

                # Handle different match types
                if "severity" in groups and groups["severity"]:
                    severity = groups["severity"]
                    message_text = groups.get("message", "")
                    flag = groups.get("flag")
                elif "In file included from" in line_stripped:
                    # This is an include chain, treat as note
                    severity = "note"
                    message_text = line_stripped
                    flag = None
                else:
                    continue

                message = CompilerMessage(
                    file_path=groups.get("file", ""),
                    line_number=int(groups["line"]) if groups.get("line") else None,
                    severity=severity,
                    message=message_text,
                    flag=flag,
                    raw_line=line,
                )

                # Make paths relative to repo root for cleaner output
                try:
                    abs_path = Path(message.file_path).resolve()
                    # Try relative to project root first, then repo root
                    try:
                        message.file_path = str(abs_path.relative_to(self.project_root))
                    except ValueError:
                        try:
                            message.file_path = str(abs_path.relative_to(self.repo_root))
                        except ValueError:
                            # Keep absolute path if we can't make it relative
                            pass
                except (ValueError, OSError):
                    pass

                self.messages.append(message)

                if message.severity == "error":
                    self.build_failed = True

        print(f"🔍 Parsed {parsed_count} compiler messages from {total_lines} lines of output")
        if self.runtime_issues:
            print(f"💥 Found {len(self.runtime_issues)} runtime issue(s)")

    def get_errors(self) -> list[CompilerMessage]:
        """Get all error messages."""
        return [msg for msg in self.messages if msg.severity == "error"]

    def get_warnings(self) -> list[CompilerMessage]:
        """Get all warning messages."""
        return [msg for msg in self.messages if msg.severity == "warning"]

    def get_summary(self) -> dict:
        """Generate a summary of the build results."""
        # Group messages by severity
        by_severity = defaultdict(list)
        for msg in self.messages:
            by_severity[msg.severity].append(msg)

        # Group warnings by flag
        warnings_by_flag = defaultdict(list)
        for msg in by_severity["warning"]:
            flag = msg.flag or "unknown"
            warnings_by_flag[flag].append(msg)

        # Group by file
        by_file = defaultdict(list)
        for msg in self.messages:
            by_file[msg.file_path].append(msg)

        # Find most common warnings
        warning_counts = defaultdict(int)
        for msg in by_severity["warning"]:
            # Normalize message for counting
            normalized = re.sub(r"\b\d+\b", "N", msg.message)
            normalized = re.sub(r"'[^']*'", "'...'", normalized)
            warning_counts[normalized] += 1

        most_common_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "build_success": not self.build_failed,
            "total_messages": len(self.messages),
            "errors": len(by_severity["error"]),
            "warnings": len(by_severity["warning"]),
            "notes": len(by_severity["note"]),
            "runtime_issues": len(self.runtime_issues),  # Add runtime issues count
            "files_with_issues": len(by_file),
            "warnings_by_flag": dict(
                sorted([(flag, len(msgs)) for flag, msgs in warnings_by_flag.items()], key=lambda x: x[1], reverse=True)
            ),
            "most_common_warnings": most_common_warnings,
            "files_with_most_issues": sorted(
                [(f, len(msgs)) for f, msgs in by_file.items()], key=lambda x: x[1], reverse=True
            )[:10],
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("METTAGRID C++ BUILD SUMMARY")
        print("=" * 80)

        if summary["build_success"]:
            print("✅ Build completed successfully")
        else:
            print("❌ Build FAILED")

        print(f"\nTotal compiler messages: {summary['total_messages']}")
        print(f"  - Errors:   {summary['errors']}")
        print(f"  - Warnings: {summary['warnings']}")
        print(f"  - Notes:    {summary['notes']}")

        if summary["runtime_issues"] > 0:
            print(f"  - Runtime issues: {summary['runtime_issues']}")

        # Print all errors with details
        errors = self.get_errors()
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            print("-" * 80)
            for i, error in enumerate(errors, 1):
                print(f"\n[Error {i}/{len(errors)}]")
                print(f"File: {error.location}")
                print(f"Message: {error.message}")
                if error.flag:
                    print(f"Flag: {error.flag}")
                print(f"Raw: {error.raw_line}")
            print("-" * 80)

        # Print runtime issues
        if self.runtime_issues:
            print(f"\n💥 RUNTIME ISSUES ({len(self.runtime_issues)}):")
            print("-" * 80)
            for i, issue in enumerate(self.runtime_issues, 1):
                print(f"[{i}] {issue}")
            print("-" * 80)

        if summary["warnings"] > 0:
            print("\nWarnings by flag:")
            for flag, count in list(summary["warnings_by_flag"].items())[:10]:
                print(f"  {flag:30} {count:5d}")

            print("\nMost common warnings:")
            for msg, count in summary["most_common_warnings"][:5]:
                print(f"  [{count:3d}x] {msg[:100]}{'...' if len(msg) > 100 else ''}")

        if summary["files_with_issues"] > 0:
            print(f"\nFiles with issues: {summary['files_with_issues']}")
            print("Top files:")
            for file_path, count in summary["files_with_most_issues"][:5]:
                print(f"  {file_path:60} {count:3d} issues")

        print("=" * 80)

    def generate_github_summary(self, title="Build Summary") -> str:
        """Generate a GitHub Actions summary in Markdown format."""
        summary = self.get_summary()
        errors = self.get_errors()

        lines = []
        lines.append(f"## 🔨 {title}\n")

        if summary["build_success"]:
            lines.append("✅ **Build Status:** Success")
        else:
            lines.append("❌ **Build Status:** FAILED")

        lines.append("\n### 📊 Statistics")
        lines.append(f"- **Total Messages:** {summary['total_messages']}")
        lines.append(f"- **Errors:** {summary['errors']}")
        lines.append(f"- **Warnings:** {summary['warnings']}")
        lines.append(f"- **Runtime Issues:** {summary['runtime_issues']}")
        lines.append(f"- **Files with Issues:** {summary['files_with_issues']}")

        # Add detailed error section
        if errors:
            lines.append("\n### ❌ Build Errors\n")
            lines.append("The following errors must be fixed:\n")

            for i, error in enumerate(errors, 1):
                lines.append(f"#### Error {i} of {len(errors)}\n")

                # Make the file path more prominent
                lines.append(f"📁 **File:** `{error.location}`\n")

                # Add a box around the error message for visibility
                lines.append("> **Error Message:**")
                lines.append("> ```")
                lines.append(f"> {error.message}")
                lines.append("> ```\n")

                if error.flag:
                    lines.append(f"🚩 **Flag:** `{error.flag}`\n")

                # Show the raw compiler output
                lines.append("**Compiler Output:**")
                lines.append("```")
                lines.append(error.raw_line)
                lines.append("```")

                # Add a horizontal rule between errors for clarity
                if i < len(errors):
                    lines.append("\n---\n")

        # Add runtime issues section
        if self.runtime_issues:
            lines.append("\n### 💥 Runtime Issues\n")
            lines.append("The following runtime issues occurred:\n")
            for i, issue in enumerate(self.runtime_issues, 1):
                lines.append(f"{i}. `{issue}`")
            lines.append("")

        if summary["warnings"] > 0:
            lines.append("\n### ⚠️ Warnings by Type")
            lines.append("| Flag | Count |")
            lines.append("|------|-------|")
            for flag, count in list(summary["warnings_by_flag"].items())[:10]:
                lines.append(f"| `{flag}` | {count} |")

            lines.append("\n### 📈 Most Common Warnings")
            lines.append("| Count | Message |")
            lines.append("|-------|---------|")
            for msg, count in summary["most_common_warnings"][:5]:
                # Escape markdown special characters
                escaped_msg = msg.replace("|", "\\|").replace("\n", " ")
                if len(escaped_msg) > 80:
                    escaped_msg = escaped_msg[:80] + "..."
                lines.append(f"| {count} | {escaped_msg} |")

        return "\n".join(lines)


def setup_build_environment(project_root: Path) -> dict:
    """Setup build environment with virtual environment if available."""
    env = os.environ.copy()

    # Virtual environment is always at repository root/.venv
    repo_root = project_root.parent
    venv_path = repo_root / ".venv"

    if venv_path.exists() and (venv_path / "bin" / "python").exists():
        print(f"📦 Using virtual environment: {venv_path}")
        # Set up environment to use the venv
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = f"{venv_path / 'bin'}:{env['PATH']}"
        # Remove any conflicting Python environment variables
        for key in ["PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"]:
            env.pop(key, None)
    else:
        print(f"⚠️  Virtual environment not found at {venv_path}")
        print("📦 Using system Python environment")

    # Force verbose output
    env["VERBOSE"] = "1"

    return env


def run_build_command(cmd: list[str], cwd: Path, env: dict) -> tuple[bool, str]:
    """Run a build command and capture output."""
    print(f"🏃 Running: {' '.join(cmd)}")
    print(f"📁 Working directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)

    # Combine stdout and stderr for analysis
    full_output = result.stdout + "\n" + result.stderr

    if not result.stdout and not result.stderr:
        print("⚠️  No output captured from build command")
    else:
        stdout_lines = len(result.stdout.splitlines()) if result.stdout else 0
        stderr_lines = len(result.stderr.splitlines()) if result.stderr else 0
        total_lines = len(full_output.splitlines())

        print(f"📝 Captured {total_lines} total lines of build output")
        print(f"📝   - stdout: {stdout_lines} lines")
        print(f"📝   - stderr: {stderr_lines} lines")

    return result.returncode == 0, full_output


def write_github_outputs(checker: BuildChecker, build_success: bool, build_output: str) -> None:
    """Write outputs for GitHub Actions."""
    if not (os.getenv("GITHUB_ACTIONS") and os.getenv("GITHUB_OUTPUT")):
        return

    summary = checker.get_summary()
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"build_success={'true' if not checker.build_failed and build_success else 'false'}\n")
        f.write(f"total_warnings={summary['warnings']}\n")
        f.write(f"total_errors={summary['errors']}\n")
        f.write(f"total_messages={summary['total_messages']}\n")
        f.write(f"runtime_issues={summary['runtime_issues']}\n")

        # Add the full build output (escaped for multiline)
        delimiter = "EOF_BUILD_OUTPUT"
        f.write(f"full_output<<{delimiter}\n")
        f.write(build_output)
        f.write(f"\n{delimiter}\n")


def write_github_summary(github_summary: str) -> None:
    """Write GitHub step summary if available."""
    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "w") as f:
            f.write(github_summary)
