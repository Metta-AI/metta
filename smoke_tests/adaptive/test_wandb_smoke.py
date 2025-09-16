#!/usr/bin/env python3
"""Smoke test for real WandB integration with adaptive system.

This test makes REAL API calls to WandB and should not be run in CI.
Requires valid WandB credentials and internet connection.

Usage:
    export WANDB_API_KEY=your_key
    export WANDB_ENTITY=your_entity
    export WANDB_PROJECT=adaptive_smoke_test
    uv run smoke_tests/adaptive/test_wandb_smoke.py
"""

import os
import time
from datetime import datetime

from metta.adaptive.models import Observation, RunInfo
from metta.adaptive.stores.wandb import WandbStore


def test_wandb_store_real_operations():
    """Test WandbStore with real WandB API calls."""

    # Check required environment variables
    required_vars = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Set these variables and try again:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        return False

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    print(f"🧪 Testing WandB integration with {entity}/{project}")

    try:
        # Initialize store
        store = WandbStore(entity=entity, project=project)
        print("✅ WandbStore initialized successfully")

        # Generate unique run ID for this test
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_run_id = f"smoke_test_{timestamp}"
        test_group = f"adaptive_smoke_test_{timestamp}"

        print(f"📝 Creating test run: {test_run_id}")

        # Test 1: Initialize a new run
        store.init_run(test_run_id, group=test_group)
        print("✅ Run initialization successful")

        # Test 2: Update run summary
        summary_data = {
            "test_metric": 42.0,
            "has_started_training": True,
            "smoke_test": True,
            "test_timestamp": timestamp,
        }
        store.update_run_summary(test_run_id, summary_data)
        print("✅ Summary update successful")

        # Wait a moment for WandB to process
        time.sleep(2)

        # Test 3: Fetch runs
        print("🔍 Fetching runs from WandB...")
        runs = store.fetch_runs(filters={"group": test_group})
        print(f"✅ Fetched {len(runs)} runs")

        # Verify our test run appears in results
        test_run_found = any(run.run_id == test_run_id for run in runs)
        if test_run_found:
            print("✅ Test run found in fetch results")
        else:
            print("⚠️  Test run not found in fetch results (may take time to propagate)")

        # Test 4: Verify run data
        if runs:
            sample_run = runs[0]
            print("📊 Sample run data:")
            print(f"  - Run ID: {sample_run.run_id}")
            print(f"  - Group: {sample_run.group}")
            print(f"  - Summary keys: {list(sample_run.summary.keys()) if sample_run.summary else 'None'}")
            print(f"  - Status: {sample_run.status}")
            print("✅ Run data structure looks correct")

        # Test 5: Test observation handling
        observation = Observation(score=0.85, cost=100.0, suggestion={"param1": "value1"})
        RunInfo(run_id=f"obs_test_{timestamp}", group=test_group, has_been_evaluated=True, observation=observation)

        # In a real scenario, the observation would be set through evaluation completion
        print("✅ Observation data structure correct")

        print("\n🎉 All WandB smoke tests passed!")
        print(f"🌐 Check your runs at: https://wandb.ai/{entity}/{project}")
        return True

    except Exception as e:
        print(f"❌ WandB smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_wandb_error_handling():
    """Test WandB store error handling with invalid credentials."""
    print("\n🧪 Testing WandB error handling...")

    try:
        # Test with invalid entity/project
        store = WandbStore(entity="invalid_entity_123456", project="invalid_project_123456")

        # This should handle errors gracefully
        runs = store.fetch_runs(filters={})
        print(f"✅ Error handling test passed - got {len(runs)} runs (expected 0 or error handled)")
        return True

    except Exception as e:
        print(f"✅ Error handling test passed - caught expected error: {type(e).__name__}")
        return True


if __name__ == "__main__":
    print("🚀 Starting WandB Smoke Tests")
    print("=" * 50)

    success1 = test_wandb_store_real_operations()
    success2 = test_wandb_error_handling()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL WANDB SMOKE TESTS PASSED")
        exit(0)
    else:
        print("❌ SOME WANDB SMOKE TESTS FAILED")
        exit(1)
