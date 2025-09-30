"""Tests for AdaptiveController - the core orchestration component."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.models import JobDefinition, JobTypes, RunInfo


class TestAdaptiveController:
    """Test the core adaptive experiment controller."""

    @pytest.fixture
    def mock_scheduler(self):
        """Mock scheduler that returns controllable job lists."""
        scheduler = Mock()
        scheduler.is_experiment_complete.return_value = False
        scheduler.schedule.return_value = []
        return scheduler

    @pytest.fixture
    def mock_dispatcher(self):
        """Mock dispatcher for job execution."""
        dispatcher = Mock()
        dispatcher.dispatch.return_value = "dispatch_id_123"
        return dispatcher

    @pytest.fixture
    def mock_store(self):
        """Mock store for run persistence."""
        store = Mock()
        store.fetch_runs.return_value = []
        return store

    @pytest.fixture
    def config(self):
        """Basic adaptive config."""
        return AdaptiveConfig(max_parallel=2, monitoring_interval=1)

    @pytest.fixture
    def controller(self, mock_scheduler, mock_dispatcher, mock_store, config):
        """Basic controller setup."""
        return AdaptiveController(
            experiment_id="test_experiment",
            scheduler=mock_scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=config,
        )

    def test_init(self, controller, mock_scheduler, mock_dispatcher, mock_store, config):
        """Test controller initialization."""
        assert controller.experiment_id == "test_experiment"
        assert controller.scheduler is mock_scheduler
        assert controller.dispatcher is mock_dispatcher
        assert controller.store is mock_store
        assert controller.config is config
        # Hooks are not stored as attributes anymore, they're passed to run()
        assert controller.dispatched_jobs == set()

    def test_experiment_completion_check(self, controller, mock_scheduler, mock_store):
        """Test that controller stops when scheduler says experiment is complete."""
        mock_scheduler.is_experiment_complete.return_value = True
        mock_store.fetch_runs.return_value = []

        # This should exit immediately without infinite loop
        controller.run()

        mock_scheduler.is_experiment_complete.assert_called_once()

    def test_job_dispatch_flow(self, controller, mock_scheduler, mock_dispatcher, mock_store):
        """Test basic job dispatch and tracking."""
        # Setup: scheduler provides one training job
        training_job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
        )
        mock_scheduler.schedule.return_value = [training_job]
        mock_store.fetch_runs.return_value = []

        # Mock experiment completion after first job
        mock_scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run()

        # Verify job was dispatched
        mock_dispatcher.dispatch.assert_called_once_with(training_job)

        # Verify run was initialized in store with initial_summary
        mock_store.init_run.assert_called_once_with("test_run_001", group="test_experiment", initial_summary={})

        # Verify job tracking
        expected_job_key = ("test_run_001", JobTypes.LAUNCH_TRAINING.value)
        assert expected_job_key in controller.dispatched_jobs

    def test_resource_constraint_validation(self, controller, mock_scheduler, mock_store):
        """Test that scheduler respects available training slots."""
        # Setup: 2 active training runs (max_parallel=2)
        active_runs = [
            RunInfo(run_id="run1", has_started_training=True, has_completed_training=False),
            RunInfo(run_id="run2", has_started_training=True, has_completed_training=False),
        ]
        mock_store.fetch_runs.return_value = active_runs

        # Set resume=True to ensure data is fetched on first iteration
        controller.config.resume = True

        # Mock experiment completion after check
        mock_scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run()

        # Scheduler should be called with 0 available slots
        mock_scheduler.schedule.assert_called_with(active_runs, 0)

    def test_eval_completion_hook(self, controller, mock_store):
        """Test that eval completion hook is called correctly."""
        hook_mock = Mock()

        # Setup: run with evaluation completed but not yet processed
        evaluated_run = RunInfo(
            run_id="test_run_001",
            has_been_evaluated=True,
            summary={},  # No processing flag yet
        )
        mock_store.fetch_runs.return_value = [evaluated_run]

        # Mock experiment completion after processing
        controller.scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run(on_eval_completed=hook_mock)

        # Verify hook was called
        hook_mock.assert_called_once_with(evaluated_run, mock_store, [evaluated_run])

        # Verify processing flag was set (check structure, not exact timestamp)
        mock_store.update_run_summary.assert_called()
        call_args = mock_store.update_run_summary.call_args
        assert call_args[0][0] == "test_run_001"
        update_dict = call_args[0][1]
        assert update_dict["adaptive/post_eval_processed"] is True
        assert "adaptive/post_eval_processed_at" in update_dict
        assert isinstance(update_dict["adaptive/post_eval_processed_at"], datetime)

    def test_job_dispatch_hook(self, controller, mock_scheduler, mock_dispatcher, mock_store):
        """Test that job dispatch hook is called after store operations."""
        dispatch_hook = Mock()

        # Setup: scheduler provides one training job
        training_job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
        )
        mock_scheduler.schedule.return_value = [training_job]
        mock_store.fetch_runs.return_value = []
        mock_scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run(on_job_dispatch=dispatch_hook)

        # Verify hook was called after dispatch and store init
        dispatch_hook.assert_called_once_with(training_job, mock_store)

    def test_duplicate_job_prevention(self, controller, mock_scheduler, mock_dispatcher, mock_store):
        """Test that duplicate jobs are not dispatched."""
        training_job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
        )

        # Pre-populate dispatched jobs
        controller.dispatched_jobs.add(("test_run_001", JobTypes.LAUNCH_TRAINING.value))

        mock_scheduler.schedule.return_value = [training_job]
        mock_store.fetch_runs.return_value = []
        mock_scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run()

        # Verify job was NOT dispatched again
        mock_dispatcher.dispatch.assert_not_called()
        mock_store.init_run.assert_not_called()

    def test_eval_job_handling(self, controller, mock_scheduler, mock_dispatcher, mock_store):
        """Test evaluation job dispatch flow."""
        eval_job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.evaluate",
            type=JobTypes.LAUNCH_EVAL,
        )

        mock_scheduler.schedule.return_value = [eval_job]
        mock_store.fetch_runs.return_value = []
        mock_scheduler.is_experiment_complete.side_effect = [False, True]

        controller.run()

        # Verify eval job was dispatched
        mock_dispatcher.dispatch.assert_called_once_with(eval_job)

        # Verify eval started flag was set (no init_run for evals)
        mock_store.init_run.assert_not_called()
        mock_store.update_run_summary.assert_called_once_with("test_run_001", {"has_started_eval": True})
