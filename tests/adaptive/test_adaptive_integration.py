"""Integration and smoke tests for adaptive experiment system."""

from unittest.mock import Mock, patch

import pytest

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler
from metta.adaptive.stores.wandb import WandbStore
from metta.tools.adaptive import AdaptiveTool, DispatcherType, SchedulerType


class TestAdaptiveIntegration:
    """Integration tests for the adaptive system components."""

    def test_controller_scheduler_store_integration(self):
        """Test that controller, scheduler, and store work together."""
        # Setup components
        config = AdaptiveConfig(max_trials=2, max_parallel=1, monitoring_interval=0.1)
        scheduler_config = TrainAndEvalConfig(max_trials=2, experiment_id="integration_test")
        scheduler = TrainAndEvalScheduler(scheduler_config)

        # Mock store and dispatcher for testing
        mock_store = Mock()
        mock_store.fetch_runs.return_value = []
        mock_dispatcher = Mock()
        mock_dispatcher.dispatch.return_value = "test_dispatch_id"

        controller = AdaptiveController(
            experiment_id="integration_test",
            scheduler=scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=config,
        )

        # Mock experiment completion after one iteration
        scheduler.is_experiment_complete = Mock(side_effect=[False, True])

        # Run the integration
        controller.run()

        # Verify integration: scheduler created jobs, controller dispatched them, store was used
        assert mock_dispatcher.dispatch.called
        assert mock_store.init_run.called
        assert scheduler.is_experiment_complete.call_count == 2

    def test_adaptive_tool_component_creation(self):
        """Test that AdaptiveTool properly creates and wires components."""
        scheduler_config = TrainAndEvalConfig(max_trials=1, experiment_id="tool_test")

        tool = AdaptiveTool(
            scheduler_type=SchedulerType.TRAIN_AND_EVAL,
            scheduler_config=scheduler_config,
            dispatcher_type=DispatcherType.LOCAL,
            config=AdaptiveConfig(max_trials=1, max_parallel=1),
            experiment_id="tool_test",
        )

        # Test scheduler creation
        scheduler = tool._create_scheduler()
        assert isinstance(scheduler, TrainAndEvalScheduler)
        assert scheduler.config.max_trials == 1

        # Test dispatcher creation
        dispatcher = tool._create_dispatcher()
        assert isinstance(dispatcher, LocalDispatcher)

    def test_hook_system_integration(self):
        """Test that the hook system works end-to-end."""
        eval_hook = Mock()
        dispatch_hook = Mock()

        config = AdaptiveConfig(max_trials=1, max_parallel=1, monitoring_interval=0.1)
        scheduler_config = TrainAndEvalConfig(max_trials=1, experiment_id="hook_test")
        scheduler = TrainAndEvalScheduler(scheduler_config)

        mock_store = Mock()
        mock_store.fetch_runs.return_value = []
        mock_dispatcher = Mock()
        mock_dispatcher.dispatch.return_value = "hook_dispatch_id"

        controller = AdaptiveController(
            experiment_id="hook_test",
            scheduler=scheduler,
            dispatcher=mock_dispatcher,
            store=mock_store,
            config=config,
            on_eval_completed=eval_hook,
            on_job_dispatch=dispatch_hook,
        )

        # Mock experiment completion after one iteration
        scheduler.is_experiment_complete = Mock(side_effect=[False, True])

        controller.run()

        # Verify hooks were wired correctly
        assert controller.on_eval_completed is eval_hook
        assert controller.on_job_dispatch is dispatch_hook

        # Verify dispatch hook was called
        assert dispatch_hook.called

    @patch("wandb.init")
    @patch("wandb.Api")
    def test_wandb_store_integration(self, mock_api, mock_init):
        """Test WandbStore integration with mocked wandb."""
        # Setup mock wandb responses
        mock_run = Mock()
        mock_run.name = "test_run"
        mock_run.summary = {}  # Make summary a real dict for item assignment
        mock_init.return_value = mock_run

        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = []
        mock_api.return_value = mock_api_instance

        # Test store operations
        store = WandbStore(entity="test_entity", project="test_project")

        # Test init_run
        store.init_run("test_run_001", group="test_group")
        mock_init.assert_called()

        # Test fetch_runs
        runs = store.fetch_runs(filters={"group": "test_group"})
        mock_api_instance.runs.assert_called()
        assert isinstance(runs, list)

    def test_sweep_agnostic_design(self):
        """Test that the adaptive system is truly sweep-agnostic."""
        # The tool should work without any sweep-specific components
        scheduler_config = TrainAndEvalConfig(max_trials=1, experiment_id="agnostic_test")

        tool = AdaptiveTool(
            scheduler_type=SchedulerType.TRAIN_AND_EVAL,
            scheduler_config=scheduler_config,
            dispatcher_type=DispatcherType.LOCAL,
            config=AdaptiveConfig(max_trials=1),
            experiment_id="agnostic_test",
            on_eval_completed=None,  # No hooks
            on_job_dispatch=None,
        )

        # Should create components without any sweep dependencies
        scheduler = tool._create_scheduler()
        dispatcher = tool._create_dispatcher()

        assert scheduler is not None
        assert dispatcher is not None

        # No sweep-specific attributes should exist on the tool
        assert not hasattr(tool, "protein_config")
        assert not hasattr(tool, "enable_sweep_observation_hook")


class TestAdaptiveSmoke:
    """Smoke tests for critical adaptive system workflows."""

    def test_local_dispatcher_basic_functionality(self):
        """Smoke test: LocalDispatcher can be created and has basic interface."""
        dispatcher = LocalDispatcher()

        # Should have dispatch method
        assert hasattr(dispatcher, "dispatch")
        assert callable(dispatcher.dispatch)

    def test_train_and_eval_scheduler_smoke(self):
        """Smoke test: TrainAndEvalScheduler core functionality."""
        config = TrainAndEvalConfig(max_trials=2, experiment_id="smoke_test")
        scheduler = TrainAndEvalScheduler(config)

        # Should be able to schedule with empty runs
        jobs = scheduler.schedule(runs=[], available_training_slots=1)
        assert len(jobs) == 1
        assert jobs[0].run_id.startswith("smoke_test_trial_")

        # Should report not complete with no runs
        assert scheduler.is_experiment_complete([]) is False

    def test_adaptive_config_validation(self):
        """Smoke test: AdaptiveConfig validation works."""
        # Valid config should not raise
        config = AdaptiveConfig(max_trials=5, max_parallel=2, monitoring_interval=60)
        config.validate()

        # Invalid configs should raise
        with pytest.raises(ValueError):
            AdaptiveConfig(max_trials=0).validate()

        with pytest.raises(ValueError):
            AdaptiveConfig(max_parallel=0).validate()

        with pytest.raises(ValueError):
            AdaptiveConfig(monitoring_interval=0).validate()

    def test_file_imports_work(self):
        """Smoke test: All key adaptive modules can be imported without errors."""
        # This test ensures our refactoring didn't break imports
        from metta.adaptive import (
            AdaptiveConfig,
            AdaptiveController,
            LocalDispatcher,
        )
        from metta.adaptive.schedulers.train_and_eval import TrainAndEvalScheduler
        from metta.tools.adaptive import AdaptiveTool

        # Basic instantiation should work
        config = AdaptiveConfig()
        tool_config = TrainAndEvalConfig(experiment_id="import_test")

        assert config is not None
        assert tool_config is not None
        assert AdaptiveController is not None
        assert LocalDispatcher is not None
        assert TrainAndEvalScheduler is not None
        assert AdaptiveTool is not None
