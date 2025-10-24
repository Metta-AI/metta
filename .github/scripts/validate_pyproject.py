#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tomli>=2.0.0; python_version < '3.11'",
# ]
# ///
"""Validate pyproject.toml files for compliance with project standards.

This script validates pyproject.toml files for:
1. Build system dependency version requirements (exact versions with ==)
2. PEP 621 compliance (license format, dynamic fields, etc.)
3. Project metadata quality and consistency

This ensures reproducible builds, prevents build failures from spec violations,
and maintains consistent project metadata across the monorepo.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class ValidationRule:
    """Represents a validation rule for pyproject.toml files."""

    name: str
    category: str
    check: Callable[[dict, Path], list[str]]
    help_text: str


# ============================================================================
# Build System Validators
# ============================================================================


def parse_requirement(req_string: str) -> tuple[str, str | None, str | None]:
    """Parse a PEP 508 requirement string into name, operator, and version.

    Returns tuple of (package_name, operator, version).
    operator and version are None if no version specifier is present.

    Handles PEP 508 requirements including:
    - Extras: package[extra]==1.0
    - Dotted names: zope.interface==6.2
    - Environment markers: package==1.0; python_version < "3.11"
    """
    req_string = req_string.strip()

    # Strip environment markers (e.g., "; python_version < '3.11'")
    if ";" in req_string:
        req_string = req_string.split(";")[0].strip()

    # Remove extras like "package[extra]"
    req_string = re.sub(r"\[.*?\]", "", req_string)

    # Match package name and version specifier
    # Package names can contain letters, numbers, underscores, hyphens, and dots (PEP 508)
    match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=~!]+)\s*(.+)$", req_string)
    if match:
        name, operator, version = match.groups()
        return name, operator, version.strip()

    # No version specifier found
    match = re.match(r"^([a-zA-Z0-9_.-]+)$", req_string)
    if match:
        return match.group(1), None, None

    return req_string, None, None


def check_exact_versions(data: dict, file_path: Path) -> list[str]:
    """Check that all build-system dependencies use exact version specifiers.

    Ensures reproducible builds by requiring exact versions (==) rather than
    version ranges (>=, >, <, <=, ~=, etc.).
    """
    errors = []
    build_system = data.get("build-system", {})
    if not build_system:
        return errors

    requires = build_system.get("requires", [])
    if not requires:
        return errors

    for req in requires:
        name, operator, version = parse_requirement(req)

        if operator is None:
            errors.append(f"  ✗ exact-versions: {req} has no version specified")
        elif operator != "==":
            errors.append(f"  ✗ exact-versions: {req} uses '{operator}' instead of '=='")

    return errors


# ============================================================================
# PEP 621 Validators
# ============================================================================


def check_license_format(data: dict, file_path: Path) -> list[str]:
    """Check that license field uses PEP 621 table format.

    PEP 621 requires:
      license = { text = "MIT" }  OR  license = { file = "LICENSE" }

    NOT:
      license = "MIT"  (old string format, not in spec)
    """
    errors = []
    project = data.get("project", {})
    if not project:
        return errors

    license_field = project.get("license")
    if license_field is None:
        return errors

    # Check if it's a string (invalid)
    if isinstance(license_field, str):
        errors.append(
            f'  ✗ license-format: license = "{license_field}" is not PEP 621 compliant\n'
            f'    → Use: license = {{ text = "{license_field}" }}\n'
            "    → See: https://peps.python.org/pep-0621/#license"
        )
        return errors

    # Check if it's a table
    if isinstance(license_field, dict):
        # Must have exactly one of 'text' or 'file'
        has_text = "text" in license_field
        has_file = "file" in license_field

        if has_text and has_file:
            errors.append(
                "  ✗ license-format: license table cannot have both 'text' and 'file' keys\n"
                '    → Use either { text = "..." } OR { file = "..." }'
            )
        elif not has_text and not has_file:
            errors.append(
                "  ✗ license-format: license table must have either 'text' or 'file' key\n"
                '    → Use: license = { text = "MIT" } or license = { file = "LICENSE" }'
            )

    return errors


def check_no_dual_specification(data: dict, file_path: Path) -> list[str]:
    """Check that fields are not specified both statically and in dynamic list.

    PEP 621: "A build back-end MUST raise an error if the metadata specifies
    a field statically as well as being listed in dynamic."
    """
    errors = []
    project = data.get("project", {})
    if not project:
        return errors

    dynamic_fields = set(project.get("dynamic", []))
    if not dynamic_fields:
        return errors

    # Check for fields that exist both statically and in dynamic
    static_fields = set(project.keys()) - {"dynamic"}
    dual_specified = static_fields & dynamic_fields

    for field in sorted(dual_specified):
        errors.append(
            f"  ✗ no-dual-specification: '{field}' is specified both statically and in dynamic list\n"
            f"    → Remove '{field}' from either the static fields or the dynamic list"
        )

    return errors


def check_name_not_dynamic(data: dict, file_path: Path) -> list[str]:
    """Check that 'name' field is not listed in dynamic.

    PEP 621: "A build back-end MUST raise an error if the metadata specifies
    name in dynamic."
    """
    errors = []
    project = data.get("project", {})
    if not project:
        return errors

    dynamic_fields = project.get("dynamic", [])
    if "name" in dynamic_fields:
        errors.append(
            "  ✗ name-not-dynamic: 'name' field cannot be in dynamic list\n"
            "    → The 'name' field must be statically defined"
        )

    return errors


def check_no_entry_point_conflicts(data: dict, file_path: Path) -> list[str]:
    """Check for deprecated entry-points.console_scripts usage.

    PEP 621: Tools "MUST raise an error if the metadata defines a
    [project.entry-points.console_scripts] or [project.entry-points.gui_scripts]
    table," as these conflict with [project.scripts] and [project.gui-scripts].
    """
    errors = []
    project = data.get("project", {})
    if not project:
        return errors

    entry_points = project.get("entry-points", {})
    if not entry_points:
        return errors

    if "console_scripts" in entry_points:
        errors.append(
            "  ✗ no-entry-point-conflicts: [project.entry-points.console_scripts] is deprecated\n"
            "    → Use [project.scripts] instead"
        )

    if "gui_scripts" in entry_points:
        errors.append(
            "  ✗ no-entry-point-conflicts: [project.entry-points.gui_scripts] is deprecated\n"
            "    → Use [project.gui-scripts] instead"
        )

    return errors


# ============================================================================
# Validation Registry
# ============================================================================

VALIDATION_RULES = [
    # Build System
    ValidationRule(
        name="exact-versions",
        category="build-system",
        check=check_exact_versions,
        help_text="Build dependencies must use exact versions (==)",
    ),
    # PEP 621
    ValidationRule(
        name="license-format",
        category="pep621",
        check=check_license_format,
        help_text="License must use PEP 621 table format",
    ),
    ValidationRule(
        name="no-dual-specification",
        category="pep621",
        check=check_no_dual_specification,
        help_text="Fields cannot be both static and in dynamic list",
    ),
    ValidationRule(
        name="name-not-dynamic",
        category="pep621",
        check=check_name_not_dynamic,
        help_text="The 'name' field cannot be in dynamic list",
    ),
    ValidationRule(
        name="no-entry-point-conflicts",
        category="pep621",
        check=check_no_entry_point_conflicts,
        help_text="Use [project.scripts] not [project.entry-points.console_scripts]",
    ),
]


# ============================================================================
# Main Validation Logic
# ============================================================================


def validate_file(file_path: Path) -> dict[str, list[str]]:
    """Validate a single pyproject.toml file.

    Returns dict mapping category names to lists of error messages.
    """
    errors_by_category = {}

    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        errors_by_category["parse-error"] = [f"Failed to parse: {e}"]
        return errors_by_category

    # Run all validators
    for rule in VALIDATION_RULES:
        errors = rule.check(data, file_path)
        if errors:
            errors_by_category.setdefault(rule.category, []).extend(errors)

    return errors_by_category


def find_pyproject_files(repo_root: Path) -> list[Path]:
    """Find all pyproject.toml files, excluding virtual environments and worktrees."""
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

    pyproject_files = []
    for path in repo_root.rglob("pyproject.toml"):
        rel_path = path.relative_to(repo_root)
        if any(part in skip_dirs for part in rel_path.parts):
            continue
        pyproject_files.append(path)

    return sorted(pyproject_files)


def format_category_name(category: str) -> str:
    """Format category name for display."""
    if category == "build-system":
        return "Build System"
    elif category == "pep621":
        return "PEP 621 Compliance"
    elif category == "parse-error":
        return "Parse Errors"
    else:
        return category.replace("-", " ").title()


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent.parent
    pyproject_files = find_pyproject_files(repo_root)

    if not pyproject_files:
        print("No pyproject.toml files found")
        return 1

    # Validate all files
    all_file_errors = {}
    total_errors = 0

    for file_path in pyproject_files:
        errors_by_category = validate_file(file_path)
        if errors_by_category:
            rel_path = file_path.relative_to(repo_root)
            all_file_errors[rel_path] = errors_by_category
            total_errors += sum(len(errors) for errors in errors_by_category.values())

    # Report results
    if all_file_errors:
        print("❌ Found pyproject.toml validation errors:\n")

        # Group by category across all files
        errors_by_category_global = {}
        for file_path, file_errors in all_file_errors.items():
            for category, errors in file_errors.items():
                errors_by_category_global.setdefault(category, []).append((file_path, errors))

        # Print errors grouped by category
        for category in sorted(errors_by_category_global.keys()):
            category_name = format_category_name(category)
            file_errors_list = errors_by_category_global[category]
            total_category_errors = sum(len(errors) for _, errors in file_errors_list)

            print("━" * 70)
            print(f" {category_name} ({total_category_errors} errors)")
            print("━" * 70)
            print()

            for file_path, errors in file_errors_list:
                print(f"{file_path}:")
                for error in errors:
                    print(error)
                print()

        print("━" * 70)
        print(f"\nSummary: {total_errors} errors in {len(all_file_errors)} files")
        print("\nFor more information:")
        print("  - Build system exact versions: See PR #3352")
        print("  - PEP 621 compliance: https://peps.python.org/pep-0621/")
        return 1

    print(f"✅ All {len(pyproject_files)} pyproject.toml files passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
