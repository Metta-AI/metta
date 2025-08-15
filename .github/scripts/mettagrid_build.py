#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Build mettagrid C++ project and check for compiler warnings and errors.
This script is designed to be run in CI to surface build issues.
"""

import argparse
import os
import re
import subprocess
import sys
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
        self.runtime_issues_list = []  # Simple list for runtime problems

    @property
    def total_warnings(self) -> int:
        """Get total number of warnings."""
        return len(self.get_warnings())

    @property
    def total_errors(self) -> int:
        """Get total number of errors."""
        return len(self.get_errors())

    @property
    def total_runtime_issues(self) -> int:
        """Get total number of runtime issues."""
        return len(self.runtime_issues_list)

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
                self.runtime_issues_list.append(line_stripped)
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
        if self.runtime_issues_list:
            print(f"💥 Found {len(self.runtime_issues_list)} runtime issue(s)")

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
            "runtime_issues": len(self.runtime_issues_list),  # Use the list length
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
        if self.runtime_issues_list:
            print(f"\n💥 RUNTIME ISSUES ({len(self.runtime_issues_list)}):")
            print("-" * 80)
            for i, issue in enumerate(self.runtime_issues_list, 1):
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
        if self.runtime_issues_list:
            lines.append("\n### 💥 Runtime Issues\n")
            lines.append("The following runtime issues occurred:\n")
            for i, issue in enumerate(self.runtime_issues_list, 1):
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


def run_mettagrid_build(
    build_type: str,
    project_root: Path,
    max_warnings: int = 50,
    clean_first: bool = True,
    with_coverage: bool = False,
    write_summary_on_success: bool = True,
) -> int:
    """
    Generic build runner for mettagrid builds.

    Args:
        build_type: Type of build ("coverage", "benchmark")
        project_root: Path to the mettagrid project root
        max_warnings: Maximum allowed warnings before failing
        clean_first: Whether to run 'make clean' before building
        with_coverage: Whether to build with coverage
        write_summary_on_success: Whether to write GitHub summary on success

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    env = setup_build_environment(project_root)

    # Clean if requested
    if clean_first:
        print("🧹 Cleaning build artifacts...")
        clean_result = subprocess.run(["make", "clean"], cwd=project_root, capture_output=True, text=True, env=env)
        if clean_result.returncode != 0:
            print(f"Warning: 'make clean' failed: {clean_result.stderr}")

    # Determine build command based on type
    if build_type == "benchmark":
        print("🔨 Building benchmarks...")
        build_cmd = ["bash", "-c", "source ../.venv/bin/activate && make benchmark VERBOSE=1"]
        title = "Benchmark Build Summary"
    else:
        # Regular build
        if with_coverage:
            print("🔨 Building project with coverage...")
            build_target = "coverage"
        else:
            print("🔨 Building project...")
            build_target = "build"
        build_cmd = ["make", build_target, "VERBOSE=1"]
        title = "Build Summary"

    # Run the build
    build_success, build_output = run_build_command(build_cmd, project_root, env)

    # Analyze the build output
    checker = BuildChecker(project_root)
    checker.parse_build_output(build_output)

    # Print summary to console
    checker.print_summary()

    # Check build quality
    print(f"\n📊 {title.replace(' Summary', '')} Quality Check (max warnings: {max_warnings})")
    print("=" * 50)

    exit_code = 0
    failure_reasons = []

    if not build_success:
        print("❌ Build command failed!")
        failure_reasons.append("Build command failed")
        exit_code = 1
    elif checker.build_failed:
        print("❌ Build completed with errors!")
        failure_reasons.append(f"Build errors: {checker.total_errors}")
        exit_code = 1
    elif checker.total_warnings > max_warnings:
        print(f"❌ Too many warnings! ({checker.total_warnings} > {max_warnings})")
        failure_reasons.append(f"Too many warnings: {checker.total_warnings} > {max_warnings}")
        exit_code = 1
    elif checker.total_runtime_issues > 0:
        print(f"❌ Runtime issues detected: {checker.total_runtime_issues}")
        failure_reasons.append(f"Runtime issues: {checker.total_runtime_issues}")
        exit_code = 1
    else:
        print("✅ Build quality check passed")

    print("=" * 50)

    # Generate GitHub summary
    github_summary = checker.generate_github_summary(title=title)

    # Add quality check result to summary
    if exit_code != 0:
        github_summary += f"\n\n### ❌ {title.replace(' Summary', '')} Quality Check Failed\n"
        for reason in failure_reasons:
            github_summary += f"- {reason}\n"
        # Always write summary on failure
        write_github_summary(github_summary)
    else:
        github_summary += f"\n\n### ✅ {title.replace(' Summary', '')} Quality Check Passed\n"
        github_summary += f"- Warnings: {checker.total_warnings}/{max_warnings}\n"
        # Only write summary on success if requested
        if write_summary_on_success:
            write_github_summary(github_summary)

    # Set outputs for GitHub Actions
    write_github_outputs(checker, build_success, build_output)

    return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check mettagrid build for compiler warnings and errors")
    parser.add_argument("build_type", choices=["coverage", "benchmark"], help="Type of build to perform")

    parser.add_argument(
        "-w", "--max-warnings", type=int, default=50, help="Maximum allowed warnings before failing (default: 50)"
    )

    args = parser.parse_args()

    # Determine project root (assumes script is in .github/scripts/)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent

    # The mettagrid subproject is in the mettagrid directory
    project_root = repo_root / "mettagrid"

    print(f"Repository root: {repo_root}")
    print(f"Mettagrid project root: {project_root}")

    # Check if we're in the right directory
    if not (project_root / "Makefile").exists():
        print(f"Error: No Makefile found in {project_root}")
        print(f"Looking for: {project_root / 'Makefile'}")
        sys.exit(1)

    # Set behavior based on build type
    if args.build_type == "benchmark":
        clean_first = False  # Benchmarks don't clean
        write_summary_on_success = False  # Only write summary on failure
        with_coverage = False  # Benchmarks don't use coverage
    else:
        clean_first = True  # Regular builds clean first
        write_summary_on_success = True  # Always write summary
        with_coverage = True  # Coverage builds use coverage

    # Run the build using the common function
    exit_code = run_mettagrid_build(
        build_type=args.build_type,
        project_root=project_root,
        max_warnings=args.max_warnings,
        clean_first=clean_first,
        with_coverage=with_coverage,
        write_summary_on_success=write_summary_on_success,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
