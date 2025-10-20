#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tomli>=2.0.0; python_version < '3.11'",
# ]
# ///
"""Check that all build-system dependencies use exact version specifiers.

This script validates that all dependencies in the [build-system] requires
section of pyproject.toml files use exact version syntax (==) rather than
version ranges (>=, >, <, <=, ~=, etc.).

This ensures reproducible builds and prevents unexpected dependency updates
from breaking the build process.
"""

import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def parse_requirement(req_string: str) -> tuple[str, str | None, str | None]:
    """Parse a PEP 508 requirement string into name, operator, and version.

    Returns tuple of (package_name, operator, version).
    operator and version are None if no version specifier is present.
    """
    # Strip whitespace
    req_string = req_string.strip()

    # Handle extras like "package[extra]"
    req_string = re.sub(r"\[.*?\]", "", req_string)

    # Match package name and version specifier
    # Pattern: package_name (operator version)
    # Package names can contain letters, numbers, underscores, hyphens, and dots (PEP 508)
    match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=~!]+)\s*(.+)$", req_string)

    if match:
        name, operator, version = match.groups()
        return name, operator, version.strip()

    # No version specifier found
    match = re.match(r"^([a-zA-Z0-9_.-]+)$", req_string)
    if match:
        return match.group(1), None, None

    # Could not parse
    return req_string, None, None


def check_exact_version(req_string: str) -> tuple[bool, str]:
    """Check if a requirement uses exact version syntax.

    Returns (is_exact, message).
    """
    name, operator, version = parse_requirement(req_string)

    if operator is None:
        return False, f"No version specified for '{name}'"

    if operator != "==":
        return False, f"Uses '{operator}' instead of '=='"

    return True, "OK"


def check_pyproject_file(file_path: Path) -> list[str]:
    """Check a single pyproject.toml file for exact version syntax.

    Returns list of error messages (empty if all checks pass).
    """
    errors = []

    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        errors.append(f"Failed to parse {file_path}: {e}")
        return errors

    # Check if [build-system] section exists
    build_system = data.get("build-system", {})
    if not build_system:
        # No build-system section, nothing to check
        return errors

    # Get requires list
    requires = build_system.get("requires", [])
    if not requires:
        # No requires, nothing to check
        return errors

    # Check each requirement
    for req in requires:
        is_exact, message = check_exact_version(req)
        if not is_exact:
            errors.append(f"  - {req}: {message}")

    return errors


def main() -> int:
    """Main entry point."""
    # Find all pyproject.toml files (excluding worktrees and virtual environments)
    repo_root = Path(__file__).parent.parent.parent
    pyproject_files = []

    # Directories to skip
    skip_dirs = {
        ".venv",
        "personal",
        "node_modules",
        ".tox",
        "venv",
        "__pycache__",
        ".bazel_output",
        "bazel-bin",
        "bazel-out",
    }

    # Find all pyproject.toml files, excluding certain directories
    for path in repo_root.rglob("pyproject.toml"):
        rel_path = path.relative_to(repo_root)
        # Skip if any parent directory is in skip_dirs
        if any(part in skip_dirs for part in rel_path.parts):
            continue
        pyproject_files.append(path)

    if not pyproject_files:
        print("No pyproject.toml files found")
        return 1

    # Check each file
    all_errors = {}
    for file_path in sorted(pyproject_files):
        errors = check_pyproject_file(file_path)
        if errors:
            rel_path = file_path.relative_to(repo_root)
            all_errors[rel_path] = errors

    # Report results
    if all_errors:
        print("❌ Found build-system dependencies with non-exact version specifiers:\n")
        for file_path, errors in all_errors.items():
            print(f"{file_path}:")
            for error in errors:
                print(error)
            print()

        print("\nAll dependencies in [build-system] requires must use exact versions (==).")
        print("This ensures reproducible builds and prevents unexpected breakage.")
        print("\nExample of correct syntax:")
        print('  requires = ["setuptools==80.9.0", "wheel==0.45.1"]')
        print("\nSee PR #3352 for the rationale behind this requirement.")
        return 1

    print(f"✅ All {len(pyproject_files)} pyproject.toml files use exact versions in [build-system] requires")
    return 0


if __name__ == "__main__":
    sys.exit(main())
