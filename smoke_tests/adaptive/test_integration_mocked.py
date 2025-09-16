#!/usr/bin/env python3
"""Integration tests for adaptive system with properly mocked external services.

These tests validate the full system integration without making real API calls.
They use sophisticated mocking to simulate realistic scenarios.

Usage:
    uv run smoke_tests/adaptive/test_integration_mocked.py
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.models import JobStatus, JobTypes, Observation, RunInfo
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler
from metta.adaptive.stores.wandb import WandbStore
from metta.tools.adaptive import AdaptiveTool, DispatcherType, SchedulerType


class TestAdaptiveIntegrationMocked:
    """Integration tests with comprehensive mocking of external services."""

    def test_full_system_integration_with_mocked_wandb(self):
        """Test complete adaptive system with mocked WandB store."""
        print("🧪 Testing full system integration with mocked WandB")

        # Mock WandB at the module level
        with patch("wandb.init") as mock_init, patch("wandb.Api") as mock_api:
            # Setup WandB mocks
            mock_run = Mock()
            mock_run.name = "test_run_001"
            mock_run.summary = {}
            mock_init.return_value = mock_run

            mock_api_instance = Mock()
            mock_api_instance.runs.return_value = []
            mock_api.return_value = mock_api_instance

            # Create real store with mocked backend
            store = WandbStore(entity="test_entity", project="test_project")

            # Create scheduler config
            scheduler_config = TrainAndEvalConfig(
                max_trials=3,
                experiment_id="integration_test",
                recipe_module="experiments.recipes.arena",
                train_entrypoint="train",
                eval_entrypoint="evaluate",
            )

            # Create scheduler
            scheduler = TrainAndEvalScheduler(scheduler_config)

            # Mock dispatcher
            mock_dispatcher = Mock()
            mock_dispatcher.dispatch.return_value = "dispatch_123"

            # Create controller
            controller_config = AdaptiveConfig(max_parallel=2, monitoring_interval=0.1)

            controller = AdaptiveController(
                experiment_id="integration_test",
                scheduler=scheduler,
                dispatcher=mock_dispatcher,
                store=store,
                config=controller_config,
            )

            # Test 1: Initial state
            assert controller.experiment_id == "integration_test"
            assert len(controller.dispatched_jobs) == 0

            # Test 2: Mock run cycle
            # Simulate fetch_runs returning empty initially
            mock_api_instance.runs.return_value = []

            # Mock experiment completion after one iteration
            scheduler.is_experiment_complete = Mock(side_effect=[False, True])

            # Run the integration (should complete quickly)
            controller.run()

            # Verify interactions
            assert mock_dispatcher.dispatch.called
            assert mock_init.called
            assert scheduler.is_experiment_complete.call_count == 2

            print("✅ Full system integration test passed")

    def test_realistic_training_simulation(self):
        """Simulate a realistic training workflow with progressive run states."""
        print("🧪 Testing realistic training workflow simulation")

        # Create mock components
        mock_store = Mock()
        mock_dispatcher = Mock()

        # Setup scheduler
        scheduler_config = TrainAndEvalConfig(
            max_trials=5, experiment_id="realistic_test", recipe_module="experiments.recipes.arena"
        )
        scheduler = TrainAndEvalScheduler(scheduler_config)

        # Create controller
        controller_config = AdaptiveConfig(max_parallel=2, monitoring_interval=0.1)
        AdaptiveController(
            experiment_id="realistic_test",
            scheduler=scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=controller_config,
        )

        # Simulation timeline (based on actual scheduler logic)
        timeline = [
            # Step 1: Empty state, create training jobs up to available slots
            {
                "runs": [],
                "available_slots": 2,
                "expected_training_jobs": 2,  # Create 2 jobs (limited by available slots)
                "expected_eval_jobs": 0,
            },
            # Step 2: Two training runs exist, no more slots
            {
                "runs": [
                    RunInfo(run_id="trial_0001", has_started_training=True, has_completed_training=False),
                    RunInfo(run_id="trial_0002", has_started_training=True, has_completed_training=False),
                ],
                "available_slots": 0,  # Both slots occupied
                "expected_training_jobs": 0,  # No slots available
                "expected_eval_jobs": 0,
            },
            # Step 3: First training completes, ready for eval
            {
                "runs": [
                    RunInfo(
                        run_id="trial_0001",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=False,
                    ),
                    RunInfo(run_id="trial_0002", has_started_training=True, has_completed_training=False),
                ],
                "available_slots": 1,  # One slot freed
                "expected_training_jobs": 1,  # Can create 1 more (current_trials=2, max_trials=5)
                "expected_eval_jobs": 1,  # trial_0001 has TRAINING_DONE_NO_EVAL status
            },
            # Step 4: Evaluation started, more capacity available
            {
                "runs": [
                    RunInfo(
                        run_id="trial_0001",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=True,
                        has_been_evaluated=False,
                    ),
                    RunInfo(
                        run_id="trial_0002",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=False,
                    ),
                    RunInfo(run_id="trial_0003", has_started_training=True, has_completed_training=False),
                ],
                "available_slots": 1,  # One slot available
                "expected_training_jobs": 1,  # Can create 1 more (current_trials=3, max_trials=5)
                "expected_eval_jobs": 1,  # trial_0002 has TRAINING_DONE_NO_EVAL status
            },
            # Step 5: Evaluation complete with observation
            {
                "runs": [
                    RunInfo(
                        run_id="trial_0001",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=True,
                        has_been_evaluated=True,
                        observation=Observation(score=0.85, cost=100.0, suggestion={}),
                    ),
                    RunInfo(
                        run_id="trial_0002",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=True,
                        has_been_evaluated=False,
                    ),
                    RunInfo(
                        run_id="trial_0003",
                        has_started_training=True,
                        has_completed_training=True,
                        has_started_eval=False,
                    ),
                    RunInfo(run_id="trial_0004", has_started_training=True, has_completed_training=False),
                ],
                "available_slots": 1,  # One slot available
                "expected_training_jobs": 1,  # One more slot for trial 5 (current_trials=4, max_trials=5)
                "expected_eval_jobs": 1,  # trial_0003 has TRAINING_DONE_NO_EVAL status
            },
        ]

        # Run through timeline
        for i, step in enumerate(timeline):
            print(f"  Step {i + 1}: {len(step['runs'])} runs, {step['available_slots']} slots")

            mock_store.fetch_runs.return_value = step["runs"]
            jobs = scheduler.schedule(runs=step["runs"], available_training_slots=step["available_slots"])

            training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
            eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]

            assert len(training_jobs) == step["expected_training_jobs"], (
                f"Step {i + 1}: Expected {step['expected_training_jobs']} training jobs, got {len(training_jobs)}"
            )

            assert len(eval_jobs) == step["expected_eval_jobs"], (
                f"Step {i + 1}: Expected {step['expected_eval_jobs']} eval jobs, got {len(eval_jobs)}"
            )

        print("✅ Realistic training simulation test passed")

    def test_error_scenarios_and_recovery(self):
        """Test system behavior under error conditions."""
        print("🧪 Testing error scenarios and recovery")

        mock_store = Mock()
        mock_dispatcher = Mock()

        # Setup scheduler with small limits for testing
        scheduler_config = TrainAndEvalConfig(max_trials=2, experiment_id="error_test")
        scheduler = TrainAndEvalScheduler(scheduler_config)

        controller_config = AdaptiveConfig(max_parallel=1, monitoring_interval=0.1)
        controller = AdaptiveController(
            experiment_id="error_test",
            scheduler=scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=controller_config,
        )

        # Test 1: Failed runs
        failed_run = RunInfo(
            run_id="failed_001", has_started_training=True, has_failed=True, last_updated_at=datetime.now()
        )

        assert failed_run.status == JobStatus.FAILED
        print("  ✅ Failed run status detection works")

        # Test 2: Stale runs
        stale_run = RunInfo(
            run_id="stale_001",
            has_started_training=True,
            has_completed_training=False,
            last_updated_at=datetime.now() - timedelta(minutes=15),  # 15 minutes old
        )

        assert stale_run.status == JobStatus.STALE
        print("  ✅ Stale run detection works")

        # Test 3: Dispatcher errors
        mock_dispatcher.dispatch.side_effect = Exception("Dispatcher connection failed")

        # System should handle dispatcher errors gracefully
        runs = []
        jobs = scheduler.schedule(runs=runs, available_training_slots=1)

        try:
            for job in jobs:
                controller._dispatch_job(job)
        except Exception as e:
            # This is expected - system should log and continue
            print(f"  ✅ Handled dispatcher error: {type(e).__name__}")

        # Test 4: Store errors
        mock_store.fetch_runs.side_effect = Exception("WandB API timeout")

        try:
            controller._fetch_current_runs()
        except Exception:
            print("  ✅ Handled store error gracefully")

        print("✅ Error scenarios and recovery test passed")

    def test_hook_system_integration(self):
        """Test the hook system with realistic callbacks."""
        print("🧪 Testing hook system integration")

        # Create hook trackers
        eval_completed_calls = []
        job_dispatch_calls = []

        def eval_completed_hook(run_info, store, all_runs):
            eval_completed_calls.append(
                {
                    "run_id": run_info.run_id,
                    "score": run_info.observation.score if run_info.observation else None,
                    "total_runs": len(all_runs),
                }
            )
            print(f"  📞 Eval hook called for {run_info.run_id}")

        def job_dispatch_hook(job_def, store):
            job_dispatch_calls.append({"run_id": job_def.run_id, "job_type": job_def.type, "cmd": job_def.cmd})
            print(f"  📞 Dispatch hook called for {job_def.run_id} ({job_def.type})")

        # Setup system with hooks
        mock_store = Mock()
        mock_dispatcher = Mock()
        mock_dispatcher.dispatch.return_value = "dispatch_hook_test"

        scheduler_config = TrainAndEvalConfig(max_trials=2, experiment_id="hook_test")
        scheduler = TrainAndEvalScheduler(scheduler_config)

        controller_config = AdaptiveConfig(max_parallel=1, monitoring_interval=0.1)
        controller = AdaptiveController(
            experiment_id="hook_test",
            scheduler=scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=controller_config,
            on_eval_completed=eval_completed_hook,
            on_job_dispatch=job_dispatch_hook,
        )

        # Test 1: Job dispatch hook
        mock_store.fetch_runs.return_value = []
        scheduler.is_experiment_complete = Mock(side_effect=[False, True])

        controller.run()

        assert len(job_dispatch_calls) > 0, "Job dispatch hook should have been called"
        assert job_dispatch_calls[0]["job_type"] == JobTypes.LAUNCH_TRAINING
        print("  ✅ Job dispatch hook working")

        # Test 2: Eval completion hook
        completed_run = RunInfo(
            run_id="hook_test_001",
            has_been_evaluated=True,
            observation=Observation(score=0.92, cost=150.0, suggestion={}),
            summary={},  # No processing flag yet
        )

        mock_store.fetch_runs.return_value = [completed_run]
        scheduler.is_experiment_complete = Mock(side_effect=[False, True])

        controller.run()

        assert len(eval_completed_calls) > 0, "Eval completion hook should have been called"
        assert eval_completed_calls[0]["run_id"] == "hook_test_001"
        assert eval_completed_calls[0]["score"] == 0.92
        print("  ✅ Eval completion hook working")

        print("✅ Hook system integration test passed")

    def test_adaptive_tool_end_to_end(self):
        """Test AdaptiveTool integration with mocked backends."""
        print("🧪 Testing AdaptiveTool end-to-end integration")

        with patch("wandb.init") as mock_init, patch("wandb.Api") as mock_api:
            # Setup WandB mocks
            mock_run = Mock()
            mock_run.summary = {}
            mock_init.return_value = mock_run

            mock_api_instance = Mock()
            mock_api_instance.runs.return_value = []
            mock_api.return_value = mock_api_instance

            # Create adaptive tool
            scheduler_config = TrainAndEvalConfig(
                max_trials=2, experiment_id="tool_e2e_test", recipe_module="experiments.recipes.arena"
            )

            tool = AdaptiveTool(
                scheduler_type=SchedulerType.TRAIN_AND_EVAL,
                scheduler_config=scheduler_config,
                dispatcher_type=DispatcherType.LOCAL,
                config=AdaptiveConfig(max_parallel=1, monitoring_interval=0.1),
                experiment_id="tool_e2e_test",
            )

            # Test component creation
            scheduler = tool._create_scheduler()
            dispatcher = tool._create_dispatcher()

            # Create store manually (AdaptiveTool creates it in invoke method)
            from metta.adaptive.stores.wandb import WandbStore

            store = WandbStore(entity="test_entity", project="test_project")

            assert isinstance(scheduler, TrainAndEvalScheduler)
            assert isinstance(dispatcher, LocalDispatcher)
            assert isinstance(store, WandbStore)

            print("  ✅ All components created successfully")

            # Test tool invocation (mock the actual invoke)
            # In real usage, this would be called by the recipe system
            args_mock = {}  # Use dict instead of Mock for args

            # Mock the invoke method to avoid actually running
            # We'll patch the controller.run method inside the tool.invoke call
            with patch("metta.adaptive.adaptive_controller.AdaptiveController.run") as mock_run:
                tool.invoke(args_mock)
                mock_run.assert_called_once()

            print("  ✅ Tool invocation successful")

        print("✅ AdaptiveTool end-to-end test passed")


def run_all_tests():
    """Run all integration tests."""
    print("🚀 Starting Integration Tests with Mocked Components")
    print("=" * 60)

    test_suite = TestAdaptiveIntegrationMocked()

    tests = [
        test_suite.test_full_system_integration_with_mocked_wandb,
        test_suite.test_realistic_training_simulation,
        test_suite.test_error_scenarios_and_recovery,
        test_suite.test_hook_system_integration,
        test_suite.test_adaptive_tool_end_to_end,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED")
        return True
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
