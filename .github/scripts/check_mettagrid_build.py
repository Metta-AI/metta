#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Check mettagrid build for compiler warnings and errors.
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
from typing import Dict, List, Optional, Tuple


@dataclass
class CompilerMessage:
    """Represents a compiler warning or error."""

    file_path: str
    line_number: Optional[int]
    severity: str  # 'warning', 'error', 'note'
    message: str
    flag: Optional[str]  # e.g., '-Wconversion'

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
        self.messages: List[CompilerMessage] = []
        self.build_failed = False

    def parse_build_output(self, output: str) -> None:
        """Parse build output and extract warnings/errors."""
        parsed_count = 0

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Try GCC/Clang pattern
            match = self.GCC_CLANG_PATTERN.match(line)
            if match:
                parsed_count += 1
                message = CompilerMessage(
                    file_path=match.group("file"),
                    line_number=int(match.group("line")),
                    severity=match.group("severity"),
                    message=match.group("message"),
                    flag=match.group("flag"),
                )

                # Make paths relative to repo root for cleaner output
                try:
                    abs_path = Path(message.file_path).resolve()
                    # Try relative to project root first, then repo root
                    try:
                        message.file_path = str(abs_path.relative_to(self.project_root))
                    except ValueError:
                        message.file_path = str(abs_path.relative_to(self.repo_root))
                except (ValueError, OSError):
                    pass

                self.messages.append(message)

                if message.severity == "error":
                    self.build_failed = True

        print(f"🔍 Parsed {parsed_count} compiler messages from output")

    def get_summary(self) -> Dict:
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
        print("BUILD SUMMARY")
        print("=" * 80)

        if summary["build_success"]:
            print("✅ Build completed successfully")
        else:
            print("❌ Build FAILED")

        print(f"\nTotal compiler messages: {summary['total_messages']}")
        print(f"  - Errors:   {summary['errors']}")
        print(f"  - Warnings: {summary['warnings']}")
        print(f"  - Notes:    {summary['notes']}")

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

    def generate_github_summary(self) -> str:
        """Generate a GitHub Actions summary in Markdown format."""
        summary = self.get_summary()

        lines = []
        lines.append("## 🔨 Build Summary\n")

        if summary["build_success"]:
            lines.append("✅ **Build Status:** Success")
        else:
            lines.append("❌ **Build Status:** FAILED")

        lines.append("\n### 📊 Statistics")
        lines.append(f"- **Total Messages:** {summary['total_messages']}")
        lines.append(f"- **Errors:** {summary['errors']}")
        lines.append(f"- **Warnings:** {summary['warnings']}")
        lines.append(f"- **Files with Issues:** {summary['files_with_issues']}")

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

        if summary["errors"] > 0:
            lines.append("\n### ❌ Errors")
            lines.append("Build failed with errors. Check the build log for details.")

        return "\n".join(lines)


def run_build(project_root: Path) -> Tuple[bool, str]:
    """Run the build process and capture output."""
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

    print("🧹 Cleaning build artifacts...")
    clean_result = subprocess.run(["make", "clean"], cwd=project_root, capture_output=True, text=True, env=env)

    if clean_result.returncode != 0:
        print(f"Warning: 'make clean' failed: {clean_result.stderr}")

    print("🔨 Building project...")
    print(f"Working directory: {project_root}")

    build_result = subprocess.run(["make", "build"], cwd=project_root, capture_output=True, text=True, env=env)

    # Combine stdout and stderr for analysis
    full_output = build_result.stdout + "\n" + build_result.stderr

    # Debug output
    if not build_result.stdout and not build_result.stderr:
        print("⚠️  No output captured from build command")
    else:
        print(f"📝 Captured {len(full_output.splitlines())} lines of build output")

    return build_result.returncode == 0, full_output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check mettagrid build for compiler warnings and errors")
    parser.add_argument("-d", "--debug", action="store_true", help="Show raw build output for debugging")

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

    # Run the build
    build_success, build_output = run_build(project_root)

    # Debug mode: show raw output
    if args.debug:
        print("\n" + "=" * 80)
        print("RAW BUILD OUTPUT")
        print("=" * 80)
        print(build_output)
        print("=" * 80 + "\n")

    # Analyze the build output
    checker = BuildChecker(project_root)
    checker.parse_build_output(build_output)

    # Print summary to console
    checker.print_summary()

    # Generate GitHub Actions summary
    github_summary = checker.generate_github_summary()

    # Write to GitHub step summary if available
    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "w") as f:
            f.write(github_summary)

    # Set outputs for GitHub Actions
    if os.getenv("GITHUB_ACTIONS") and os.getenv("GITHUB_OUTPUT"):
        summary = checker.get_summary()
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"build_success={'true' if not checker.build_failed and build_success else 'false'}\n")
            f.write(f"total_warnings={summary['warnings']}\n")
            f.write(f"total_errors={summary['errors']}\n")
            f.write(f"total_messages={summary['total_messages']}\n")

    # Exit with appropriate code - only fail on actual build failure or errors
    if not build_success:
        print("\n❌ Build command failed!")
        sys.exit(1)
    elif checker.build_failed:
        print("\n❌ Build completed with errors!")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
