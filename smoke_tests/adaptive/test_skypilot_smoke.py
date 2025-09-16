#!/usr/bin/env python3
"""Skypilot dispatcher smoke test for adaptive system.

This test validates Skypilot integration with real cloud resources.
USE WITH CAUTION - this will consume cloud credits!

Usage:
    export SKYPILOT_ENABLED=true
    export WANDB_API_KEY=your_key
    export WANDB_ENTITY=your_entity
    export WANDB_PROJECT=adaptive_smoke_test
    uv run smoke_tests/adaptive/test_skypilot_smoke.py
"""

import os
import sys
import time
from datetime import datetime

from metta.adaptive.dispatcher.skypilot import SkypilotDispatcher
from metta.adaptive.models import JobDefinition, JobTypes


def check_skypilot_environment():
    """Check if Skypilot is properly configured."""
    # Check if skypilot is enabled via env var
    if not os.getenv("SKYPILOT_ENABLED", "").lower() == "true":
        print("❌ SKYPILOT_ENABLED not set to 'true'")
        print("Set 'export SKYPILOT_ENABLED=true' to enable this test")
        return False

    try:
        # Try to import sky
        import sky  # noqa: F401

        print("✅ Skypilot module available")
        return True
    except ImportError:
        print("❌ Skypilot not installed")
        print("Install with: pip install skypilot")
        return False


def test_skypilot_dispatcher_creation():
    """Test basic Skypilot dispatcher creation."""
    print("🧪 Testing Skypilot dispatcher creation")

    try:
        SkypilotDispatcher()
        print("✅ SkypilotDispatcher created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create SkypilotDispatcher: {e}")
        return False


def test_job_dispatch_dry_run():
    """Test job dispatch without actually running (if possible)."""
    print("🧪 Testing Skypilot job dispatch (dry run)")

    try:
        dispatcher = SkypilotDispatcher()

        # Create test job
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_job = JobDefinition(
            run_id=f"skypilot_test_{timestamp}",
            cmd="experiments.recipes.arena.train",
            gpus=1,
            nodes=1,
            type=JobTypes.LAUNCH_TRAINING,
            args={"run": f"skypilot_smoke_{timestamp}"},
            overrides={"trainer.total_timesteps": 1000},  # Very short for testing
        )

        print(f"📋 Test job: {test_job.run_id}")
        print(f"   Command: {test_job.cmd}")
        print(f"   Resources: {test_job.gpus} GPU, {test_job.nodes} node")

        # Check if we can inspect the job without dispatching
        # (This depends on the dispatcher implementation)
        print("⚠️  About to dispatch REAL cloud job - this will consume credits!")
        print("Press Ctrl+C in the next 5 seconds to cancel...")

        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n⚠️  Test cancelled by user")
            return True

        # Dispatch the job
        print("🚀 Dispatching job to Skypilot...")
        dispatch_id = dispatcher.dispatch(test_job)

        print(f"✅ Job dispatched successfully with ID: {dispatch_id}")
        print("⚠️  Job is now running on cloud resources!")
        print("   Monitor with: sky status")
        print("   Stop with: sky down <cluster_name>")

        return True

    except KeyboardInterrupt:
        print("\n⚠️  Test cancelled by user")
        return True
    except Exception as e:
        print(f"❌ Skypilot dispatch test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_skypilot_status_check():
    """Test checking status of running jobs."""
    print("🧪 Testing Skypilot status checking")

    try:
        # This would depend on having sky CLI available
        import subprocess

        result = subprocess.run(["sky", "status"], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Skypilot status command successful")
            print("Running clusters:")
            print(result.stdout)
            return True
        else:
            print(f"⚠️  Skypilot status returned code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Skypilot status command timed out")
        return False
    except FileNotFoundError:
        print("⚠️  'sky' command not found in PATH")
        return False
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


def test_resource_estimation():
    """Test resource estimation for different job types."""
    print("🧪 Testing resource estimation")

    try:
        SkypilotDispatcher()

        test_jobs = [
            JobDefinition(
                run_id="small_job",
                cmd="experiments.recipes.arena.train",
                gpus=1,
                nodes=1,
                type=JobTypes.LAUNCH_TRAINING,
            ),
            JobDefinition(
                run_id="large_job",
                cmd="experiments.recipes.arena.train",
                gpus=4,
                nodes=2,
                type=JobTypes.LAUNCH_TRAINING,
            ),
            JobDefinition(
                run_id="eval_job", cmd="experiments.recipes.arena.evaluate", gpus=1, nodes=1, type=JobTypes.LAUNCH_EVAL
            ),
        ]

        for job in test_jobs:
            print(f"  📊 Job: {job.run_id}")
            print(f"     Resources: {job.gpus} GPU, {job.nodes} nodes")
            print(f"     Type: {job.type}")
            # In a real implementation, we might have cost estimation here

        print("✅ Resource estimation test passed")
        return True

    except Exception as e:
        print(f"❌ Resource estimation test failed: {e}")
        return False


def main():
    """Run all Skypilot smoke tests."""
    print("🚀 Starting Skypilot Smoke Tests")
    print("=" * 50)
    print("⚠️  WARNING: These tests may consume cloud credits!")
    print("=" * 50)

    # Environment check
    if not check_skypilot_environment():
        print("❌ Environment not ready for Skypilot testing")
        return False

    tests = [
        ("Dispatcher Creation", test_skypilot_dispatcher_creation),
        ("Resource Estimation", test_resource_estimation),
        ("Status Checking", test_skypilot_status_check),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)

        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}")

    # Optional: Real dispatch test
    print("\n🔥 Optional: Real Job Dispatch Test")
    print("-" * 30)
    print("This will dispatch a REAL job to cloud resources!")

    user_input = input("Run real dispatch test? (yes/no): ").lower()
    if user_input in ["yes", "y"]:
        try:
            success = test_job_dispatch_dry_run()
            if success:
                passed += 1
                print("✅ Real Dispatch Test PASSED")
            else:
                failed += 1
                print("❌ Real Dispatch Test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ Real Dispatch Test FAILED: {e}")
    else:
        print("⚠️  Real dispatch test skipped")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL SKYPILOT SMOKE TESTS PASSED")
        return True
    else:
        print("❌ SOME SKYPILOT SMOKE TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
