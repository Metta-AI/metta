#!/usr/bin/env python3
"""Script to identify and fix mid-file imports that can be moved to the top."""

import re
import subprocess

# Patterns that indicate the import should stay where it is
SKIP_PATTERNS = [
    r"if\s+TYPE_CHECKING:",  # TYPE_CHECKING blocks
    r"try:",  # Try/except blocks
    r"#.*circular",  # Comments mentioning circular imports
    r"#.*avoid.*import",  # Comments about avoiding imports
    r"#.*noqa",  # noqa comments
    r"#.*pragma:",  # pragma comments
    r"import\s+importlib",  # importlib dynamic imports
]


def should_skip_context(lines, import_line_idx):
    """Check if the import is in a context that should be skipped."""
    # Check 5 lines before the import
    start = max(0, import_line_idx - 5)
    context = "\n".join(lines[start : import_line_idx + 2])

    for pattern in SKIP_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def analyze_file(filepath):
    """Analyze a Python file for imports that could be moved to the top."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    issues = []
    for i, line in enumerate(lines):
        # Skip if not an indented import
        if not re.match(r"^[ \t]+(from .+ import|import )", line):
            continue

        # Skip if in a context that should be left alone
        if should_skip_context(lines, i):
            continue

        issues.append(
            {"line_num": i + 1, "line": line.strip(), "context": lines[max(0, i - 2) : min(len(lines), i + 3)]}
        )

    return issues


def main():
    # Get all Python files with indented imports
    result = subprocess.run(
        [
            "grep",
            "-l",
            "-r",
            "--include=*.py",
            "-E",
            r"^[ \t]+(from .+ import|import )",
            "/Users/daveey/code/metta/metta",
            "/Users/daveey/code/metta/agent",
            "/Users/daveey/code/metta/app_backend",
            "/Users/daveey/code/metta/common",
            "/Users/daveey/code/metta/packages/cogames",
            "/Users/daveey/code/metta/packages/mettagrid/python",
        ],
        capture_output=True,
        text=True,
    )

    files = [f for f in result.stdout.strip().split("\n") if f]

    print(f"Found {len(files)} files with indented imports")
    print("\nAnalyzing files...\n")

    fixable_files = []
    for filepath in files:
        issues = analyze_file(filepath)
        if issues:
            print(f"\n{filepath}:")
            for issue in issues:
                print(f"  Line {issue['line_num']}: {issue['line']}")
            fixable_files.append((filepath, issues))

    print(f"\n\nTotal files with potential fixes: {len(fixable_files)}")


if __name__ == "__main__":
    main()
