#!/usr/bin/env python3
"""Convenience script to run all adaptive smoke tests.

Usage:
    # Set environment variables first
    export WANDB_API_KEY=your_key
    export WANDB_ENTITY=your_entity
    export WANDB_PROJECT=adaptive_smoke_test

    # Optional for Skypilot tests
    export SKYPILOT_ENABLED=true

    # Run all tests
    uv run smoke_tests/adaptive/run_all.py

    # Run specific categories
    uv run smoke_tests/adaptive/run_all.py --integration-only
    uv run smoke_tests/adaptive/run_all.py --smoke-only
    uv run smoke_tests/adaptive/run_all.py --safe-only  # No real API calls
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, description: str) -> bool:
    """Run a test script and return success status."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Running: {description}")
    print(f"📄 Script: {script_path}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], cwd=script_path.parent.parent.parent
        )  # Run from repo root

        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED (exit code: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ {description} FAILED with exception: {e}")
        return False


def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment...")

    # Required for most tests
    required_vars = ["WANDB_API_KEY", "WANDB_ENTITY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("Some tests may fail without these variables.")
    else:
        print("✅ Core environment variables set")

    # Optional variables
    optional_vars = {
        "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "adaptive_smoke_test"),
        "SKYPILOT_ENABLED": os.getenv("SKYPILOT_ENABLED", "false"),
    }

    print("📋 Environment configuration:")
    for var, value in optional_vars.items():
        print(f"   {var}: {value}")

    return len(missing_vars) == 0


def main():
    parser = argparse.ArgumentParser(description="Run adaptive smoke tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests (with mocks)")
    parser.add_argument("--smoke-only", action="store_true", help="Run only real API smoke tests")
    parser.add_argument("--safe-only", action="store_true", help="Run only tests that don't make real API calls")
    parser.add_argument("--skip-skypilot", action="store_true", help="Skip Skypilot tests (they consume cloud credits)")

    args = parser.parse_args()

    print("🚀 Adaptive System Smoke Test Runner")

    # Check environment
    if not check_environment():
        print("⚠️  Environment issues detected, some tests may fail")

    # Define test categories
    smoke_tests_dir = Path(__file__).parent

    test_categories = {
        "safe": [(smoke_tests_dir / "test_integration_mocked.py", "Integration Tests (Mocked)")],
        "smoke": [
            (smoke_tests_dir / "test_wandb_smoke.py", "WandB Smoke Test"),
            (smoke_tests_dir / "test_full_workflow_smoke.py", "Full Workflow Smoke Test"),
        ],
        "cloud": [(smoke_tests_dir / "test_skypilot_smoke.py", "Skypilot Smoke Test")],
    }

    # Determine which tests to run
    tests_to_run = []

    if args.safe_only:
        tests_to_run.extend(test_categories["safe"])
    elif args.integration_only:
        tests_to_run.extend(test_categories["safe"])
    elif args.smoke_only:
        tests_to_run.extend(test_categories["smoke"])
    else:
        # Run all by default
        tests_to_run.extend(test_categories["safe"])
        tests_to_run.extend(test_categories["smoke"])
        if not args.skip_skypilot:
            tests_to_run.extend(test_categories["cloud"])

    if not tests_to_run:
        print("❌ No tests selected")
        return False

    print(f"\n📋 Selected {len(tests_to_run)} test suite(s):")
    for _, description in tests_to_run:
        print(f"   • {description}")

    # Run tests
    results = []
    passed = 0
    failed = 0

    for script_path, description in tests_to_run:
        success = run_script(script_path, description)
        results.append((description, success))

        if success:
            passed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 FINAL RESULTS")
    print(f"{'=' * 60}")

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {description}")

    print(f"\n📈 Summary: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL SMOKE TESTS PASSED!")
        return True
    else:
        print("❌ SOME SMOKE TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
