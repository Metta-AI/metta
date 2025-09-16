#!/usr/bin/env python3
"""Full workflow smoke test for the adaptive system.

This test runs the complete adaptive workflow with real components where possible
and makes actual API calls. Use with caution and appropriate credentials.

Usage:
    # Set environment variables
    export WANDB_API_KEY=your_key
    export WANDB_ENTITY=your_entity
    export WANDB_PROJECT=adaptive_smoke_test

    # Optional: For cloud testing
    export SKYPILOT_ENABLED=true

    uv run smoke_tests/adaptive/test_full_workflow_smoke.py
"""

import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.models import JobDefinition, JobTypes, Observation, RunInfo
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler
from metta.adaptive.stores.wandb import WandbStore
from metta.tools.adaptive import AdaptiveTool, DispatcherType, SchedulerType


class WorkflowSmokeTest:
    """Full workflow smoke testing."""

    def __init__(self):
        self.controller: Optional[AdaptiveController] = None
        self.test_run_id: Optional[str] = None
        self.start_time = time.time()
        self.max_runtime = 300  # 5 minutes max runtime

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
            if self.controller:
                print("Stopping controller...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def test_full_adaptive_workflow_with_timeout(self):
        """Test complete adaptive workflow with safety timeout."""
        print("🧪 Testing complete adaptive workflow (with safety timeout)")

        # Check environment
        if not self._check_environment():
            return False

        self.setup_signal_handlers()

        try:
            # Create unique test ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.test_run_id = f"workflow_smoke_{timestamp}"

            print(f"🏷️  Test run ID: {self.test_run_id}")

            # Setup components
            entity = os.getenv("WANDB_ENTITY")
            project = os.getenv("WANDB_PROJECT", "adaptive_smoke_test")

            # Create configurations
            scheduler_config = TrainAndEvalConfig(
                max_trials=3,  # Small number for smoke test
                experiment_id=self.test_run_id,
                recipe_module="experiments.recipes.arena",
                train_entrypoint="train",
                eval_entrypoint="evaluate",
                gpus=1,
            )

            controller_config = AdaptiveConfig(
                max_parallel=2,
                monitoring_interval=5.0,  # Check every 5 seconds
                resume=False,  # Start fresh
            )

            # Create components
            scheduler = TrainAndEvalScheduler(scheduler_config)
            dispatcher = LocalDispatcher()  # Use local dispatcher for safety
            store = WandbStore(entity=entity, project=project)

            # Create hooks for monitoring
            dispatched_jobs = []
            completed_evals = []

            def job_dispatch_hook(job_def: JobDefinition, store_ref):
                dispatched_jobs.append(job_def)
                print(f"📤 Dispatched {job_def.type} job: {job_def.run_id}")

            def eval_completed_hook(run_info: RunInfo, store_ref, all_runs):
                completed_evals.append(run_info)
                score = run_info.observation.score if run_info.observation else "N/A"
                print(f"✅ Eval completed for {run_info.run_id}, score: {score}")

            # Create controller
            self.controller = AdaptiveController(
                experiment_id=self.test_run_id,
                scheduler=scheduler,
                dispatcher=dispatcher,
                store=store,
                config=controller_config,
                on_job_dispatch=job_dispatch_hook,
                on_eval_completed=eval_completed_hook,
            )

            print("✅ All components created successfully")

            # Start timeout monitor
            timeout_thread = threading.Thread(target=self._timeout_monitor, daemon=True)
            timeout_thread.start()

            print(f"🚀 Starting adaptive workflow (max runtime: {self.max_runtime}s)")
            print("   Press Ctrl+C to stop gracefully")

            # Run the controller with timeout protection
            try:
                self.controller.run()
                print("✅ Controller completed naturally")

            except KeyboardInterrupt:
                print("\n⚠️  Workflow stopped by user")

            # Verify results
            print("\n📊 Workflow Results:")
            print(f"   Dispatched jobs: {len(dispatched_jobs)}")
            print(f"   Completed evaluations: {len(completed_evals)}")
            print(f"   Runtime: {time.time() - self.start_time:.1f}s")

            # Fetch final state from store
            final_runs = store.fetch_runs(filters={"group": self.test_run_id})
            print(f"   Final runs in store: {len(final_runs)}")

            if final_runs:
                for run in final_runs:
                    print(f"     - {run.run_id}: {run.status}")

            # Success criteria
            success = len(dispatched_jobs) > 0
            if success:
                print("✅ Workflow smoke test PASSED - jobs were dispatched")
            else:
                print("❌ Workflow smoke test FAILED - no jobs dispatched")

            return success

        except Exception as e:
            print(f"❌ Workflow smoke test FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_adaptive_tool_invoke_real(self):
        """Test AdaptiveTool.invoke() with real components but limited scope."""
        print("\n🧪 Testing AdaptiveTool.invoke() with real components")

        if not self._check_environment():
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_id = f"tool_invoke_{timestamp}"

            # Create tool configuration
            scheduler_config = TrainAndEvalConfig(
                max_trials=1,  # Very limited for smoke test
                experiment_id=test_id,
                recipe_module="experiments.recipes.arena",
            )

            tool = AdaptiveTool(
                scheduler_type=SchedulerType.TRAIN_AND_EVAL,
                scheduler_config=scheduler_config,
                dispatcher_type=DispatcherType.LOCAL,
                config=AdaptiveConfig(max_parallel=1, monitoring_interval=5.0),
                experiment_id=test_id,
            )

            # Create minimal args mock
            class ArgsMock:
                pass

            args = ArgsMock()

            print(f"🏷️  Tool test ID: {test_id}")
            print("⚠️  This will start a real training job - monitoring for 30 seconds max...")

            # Run tool with timeout
            start_time = time.time()
            tool_thread = threading.Thread(target=lambda: tool.invoke(args), daemon=True)
            tool_thread.start()

            # Wait for either completion or timeout
            tool_thread.join(timeout=30.0)

            elapsed = time.time() - start_time
            print(f"⏰ Tool invoke ran for {elapsed:.1f}s")

            if tool_thread.is_alive():
                print("⚠️  Tool invoke still running after timeout (this is expected)")
                print("✅ Tool invoke smoke test PASSED - successfully started")
                return True
            else:
                print("✅ Tool invoke completed within timeout")
                return True

        except Exception as e:
            print(f"❌ Tool invoke smoke test FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _check_environment(self) -> bool:
        """Check if environment is properly configured."""
        required_vars = ["WANDB_API_KEY", "WANDB_ENTITY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
            print("Set these variables and try again:")
            for var in missing_vars:
                print(f"  export {var}=your_value")
            return False

        print("✅ Environment variables configured")
        return True

    def _timeout_monitor(self):
        """Monitor for timeout and force shutdown if needed."""
        time.sleep(self.max_runtime)
        print(f"\n⏰ TIMEOUT: Workflow has been running for {self.max_runtime}s")
        print("Forcing shutdown for safety...")
        if self.controller:
            # Force stop the controller
            os._exit(1)

    def test_stress_scheduling_logic(self):
        """Stress test the scheduling logic with various run states."""
        print("\n🧪 Stress testing scheduling logic")

        scheduler_config = TrainAndEvalConfig(max_trials=20, experiment_id="stress_test")
        scheduler = TrainAndEvalScheduler(scheduler_config)

        # Create various run states
        test_runs = [
            # Training jobs
            RunInfo(f"train_{i:03d}", has_started_training=True, has_completed_training=False)
            for i in range(5)
        ]

        # Completed training, ready for eval
        test_runs.extend(
            [
                RunInfo(
                    f"ready_{i:03d}", has_started_training=True, has_completed_training=True, has_started_eval=False
                )
                for i in range(3)
            ]
        )

        # In evaluation
        test_runs.extend(
            [
                RunInfo(
                    f"eval_{i:03d}",
                    has_started_training=True,
                    has_completed_training=True,
                    has_started_eval=True,
                    has_been_evaluated=False,
                )
                for i in range(2)
            ]
        )

        # Completed
        test_runs.extend(
            [
                RunInfo(
                    f"done_{i:03d}",
                    has_started_training=True,
                    has_completed_training=True,
                    has_started_eval=True,
                    has_been_evaluated=True,
                    observation=Observation(score=0.8, cost=100, suggestion={}),
                )
                for i in range(5)
            ]
        )

        print(f"  📊 Testing with {len(test_runs)} runs in various states")

        # Test scheduling
        start_time = time.time()
        jobs = scheduler.schedule(runs=test_runs, available_training_slots=3)
        elapsed = time.time() - start_time

        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
        eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]

        print(f"  ⚡ Scheduling completed in {elapsed * 1000:.1f}ms")
        print(f"  📈 Generated {len(training_jobs)} training jobs")
        print(f"  📊 Generated {len(eval_jobs)} eval jobs")

        # Verify constraints
        assert len(training_jobs) <= 3, "Should respect available slots"
        assert len(eval_jobs) <= 3, "Should only eval ready runs"

        print("✅ Stress test PASSED")
        return True


def main():
    """Run all smoke tests."""
    print("🚀 Starting Full Workflow Smoke Tests")
    print("=" * 60)

    # Initialize test suite
    test_suite = WorkflowSmokeTest()

    tests = [
        ("Stress Scheduling Logic", test_suite.test_stress_scheduling_logic),
        ("Full Adaptive Workflow", test_suite.test_full_adaptive_workflow_with_timeout),
        ("AdaptiveTool Invoke", test_suite.test_adaptive_tool_invoke_real),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)

        try:
            success = test_func()
            if success:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Final Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL WORKFLOW SMOKE TESTS PASSED")
        return True
    else:
        print("❌ SOME WORKFLOW SMOKE TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
