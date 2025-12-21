#!/usr/bin/env python3
"""CI validation script for Python import patterns.

This draft script is intended for planned integration into CI.

Usage:
    python tools/dev/python_imports/ci_validate_imports.py [--strict]

Exit codes:
    0 - No violations (or only warnings in gradual mode)
    1 - Violations found that should fail CI

Integration Plan:
    - Add to metta ci command
    - Run on every PR
    - Initially in gradual mode
    - Switch to strict mode after all packages migrated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_circular_dependencies(repo_root: Path, strict: bool = False) -> tuple[bool, str]:
    """Check for circular dependencies.

    Args:
        repo_root: Root of the repository
        strict: If True, fail on any cycles. If False, only fail on cross-package cycles.

    Returns:
        (passed, message) tuple
    """
    cycles_report = repo_root / "tools/dev/python_imports/cycles_report.json"

    if not cycles_report.exists():
        return False, "❌ Cycles report not found. Run detect_cycles.py first."

    with open(cycles_report) as f:
        data = json.load(f)

    summary = data["summary"]
    cross_package = summary["cross_package_cycles"]
    same_package = summary["same_package_cycles"]

    messages = []
    passed = True

    # Cross-package cycles always fail (architectural violations)
    if cross_package > 0:
        passed = False
        messages.append(f"FAIL: {cross_package} cross-package circular dependencies found")
        messages.append("   Cross-package cycles violate architecture layers and must be fixed.")

        for cycle in data["cycles"]["cross_package_cycles"]:
            packages = cycle["packages"]
            modules = cycle["modules"]
            messages.append(f"   - {' ↔ '.join(packages)}")
            messages.append(f"     Modules: {' → '.join(modules[:3])}...")

    # Same-package cycles: fail in strict mode, warn in gradual mode
    if same_package > 0:
        if strict:
            passed = False
            messages.append(f"FAIL: {same_package} same-package circular dependencies found (strict mode)")
        else:
            messages.append(f"WARN: {same_package} same-package circular dependencies found")
            messages.append("   These should be resolved but won't fail CI in gradual mode.")

        affected_packages = set()
        for cycle in data["cycles"]["same_package_cycles"]:
            affected_packages.update(cycle["packages"])

        # Filter out .bazel_output cycles
        real_packages = {p for p in affected_packages if not p.startswith("packages.mettagrid..bazel")}

        if real_packages:
            messages.append(f"   Affected packages: {', '.join(sorted(real_packages))}")

    if passed and cross_package == 0 and same_package == 0:
        messages.append("PASS: No circular dependencies found")

    return passed, "\n".join(messages)


def check_layer_violations(repo_root: Path) -> tuple[bool, str]:
    """Check for architecture layer violations.

    This is a placeholder for planned implementation.
    Will check that higher layers don't import from lower layers.

    Returns:
        (passed, message) tuple
    """
    # TODO:
    # Would analyze imports to ensure:
    # - common doesn't import from anything
    # - mettagrid doesn't import from agent/metta
    # - agent doesn't import from metta.rl
    # etc.

    return True, "SKIP: Layer validation not yet implemented"


def check_type_checking_usage(repo_root: Path) -> tuple[bool, str]:
    """Check for unjustified TYPE_CHECKING usage.

    This is a placeholder for planned implementation.
    Will ensure TYPE_CHECKING is only used with documented justification.

    Returns:
        (passed, message) tuple
    """
    # TODO:
    # Would check:
    # - TYPE_CHECKING blocks have justification comments
    # - No new TYPE_CHECKING without exception approval

    return True, "SKIP: TYPE_CHECKING validation not yet implemented"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI validation for Python import patterns (draft)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Modes:
  --gradual (default): Fail on cross-package cycles only, warn on same-package
  --strict:            Fail on any circular dependencies

Planned Integration:
  This script will be integrated into `metta ci` command
  Run after every code change to catch import violations early

Current Status: DRAFT - Not yet integrated into CI
        """,
    )
    parser.add_argument(
        "--strict", action="store_true", help="Strict mode: fail on any cycles (use after full migration)"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="Repository root path (default: current directory)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Python Import Pattern Validation (Draft)")
    print("=" * 70)
    print()

    mode = "STRICT" if args.strict else "GRADUAL"
    print(f"Mode: {mode}")
    print()

    all_passed = True

    # Check 1: Circular dependencies
    print("Check 1: Circular Dependencies")
    print("-" * 70)
    passed, message = check_circular_dependencies(args.repo_root, args.strict)
    print(message)
    print()
    all_passed = all_passed and passed

    # Check 2: Layer violations (TODO)
    print("Check 2: Architecture Layer Violations")
    print("-" * 70)
    passed, message = check_layer_violations(args.repo_root)
    print(message)
    print()
    all_passed = all_passed and passed

    # Check 3: TYPE_CHECKING usage (TODO)
    print("Check 3: TYPE_CHECKING Usage")
    print("-" * 70)
    passed, message = check_type_checking_usage(args.repo_root)
    print(message)
    print()
    all_passed = all_passed and passed

    # Summary
    print("=" * 70)
    if all_passed:
        print("All checks passed")
        print()
        print("Note: This is a draft script.")
        print("Not yet integrated into CI - for testing only.")
        return 0
    else:
        print("Some checks failed")
        print()
        print("Note: This is a draft script.")
        print("These failures would block CI once integrated.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
