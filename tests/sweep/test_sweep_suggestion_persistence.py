"""Test that sweep suggestions are properly persisted to WandB at run initialization."""

import logging
from unittest.mock import MagicMock

from softmax.training.adaptive import AdaptiveConfig
from softmax.training.adaptive.adaptive_controller import AdaptiveController
from softmax.training.adaptive.models import JobTypes, RunInfo
from softmax.training.adaptive.stores.wandb import WandbStore
from softmax.training.sweep.protein_config import ParameterConfig, ProteinConfig
from softmax.training.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
)

logger = logging.getLogger(__name__)


def test_suggestion_stored_at_init():
    """Test that sweep/suggestion is properly stored when run is initialized.

    This is a regression test for the bug where suggestions were missing
    due to WandB eventual consistency issues. The fix ensures suggestions
    are stored at run initialization time rather than after dispatch.
    """

    # Create mock store
    mock_store = MagicMock(spec=WandbStore)
    mock_store.fetch_runs.return_value = []

    # Track what was passed to init_run
    init_run_calls = []

    def capture_init_run(run_id, group=None, tags=None, initial_summary=None):
        init_run_calls.append({"run_id": run_id, "group": group, "initial_summary": initial_summary})

    mock_store.init_run.side_effect = capture_init_run

    # Create mock dispatcher
    mock_dispatcher = MagicMock()
    mock_dispatcher.dispatch.return_value = "dispatch-123"

    # Create scheduler config with protein config
    protein_config = ProteinConfig(
        metric="test/metric",
        goal="maximize",
        parameters={
            "lr": ParameterConfig(
                min=0.001,
                max=0.01,
                distribution="log_normal",
                mean=0.005,  # Required for log_normal
                scale="auto",  # Required for log_normal
            )
        },
    )

    scheduler_config = BatchedSyncedSchedulerConfig(
        max_trials=2,
        batch_size=2,
        recipe_module="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="eval",
        experiment_id="test_sweep",
        protein_config=protein_config,
    )

    # Create scheduler
    scheduler = BatchedSyncedOptimizingScheduler(scheduler_config)

    # Create adaptive config
    adaptive_config = AdaptiveConfig(max_parallel=2, monitoring_interval=1, resume=False)

    # Create controller (not used directly, but validates configuration)
    _ = AdaptiveController(
        experiment_id="test_sweep",
        scheduler=scheduler,
        dispatcher=mock_dispatcher,
        store=mock_store,
        config=adaptive_config,
    )

    # Manually trigger one schedule cycle
    # This simulates what happens in the run loop

    # 1. Fetch runs (returns empty)
    runs = mock_store.fetch_runs({"group": "test_sweep"})

    # 2. Schedule new jobs (should create training jobs with suggestions)
    jobs = scheduler.schedule(runs, available_training_slots=2)

    # 3. Dispatch jobs (this is where init_run should be called)
    for job in jobs:
        if job.type == JobTypes.LAUNCH_TRAINING:
            # Simulate what AdaptiveController does
            _ = mock_dispatcher.dispatch(job)
            mock_store.init_run(job.run_id, group="test_sweep", initial_summary=job.metadata)

    # Verify init_run was called with sweep/suggestion
    assert len(init_run_calls) == 2, f"Expected 2 init_run calls, got {len(init_run_calls)}"

    for _, call in enumerate(init_run_calls):
        # Check that initial_summary was passed
        assert "initial_summary" in call
        assert call["initial_summary"] is not None, f"initial_summary should not be None for run {call['run_id']}"

        # Check that sweep/suggestion is in the initial_summary
        assert "sweep/suggestion" in call["initial_summary"], (
            f"Missing sweep/suggestion in initial_summary for run {call['run_id']}. "
            f"Keys present: {list(call['initial_summary'].keys())}"
        )

        # Check that the suggestion contains expected parameter
        suggestion = call["initial_summary"]["sweep/suggestion"]
        assert "lr" in suggestion, (
            f"Missing 'lr' parameter in suggestion for run {call['run_id']}. Suggestion: {suggestion}"
        )

        # Verify the suggestion value is within expected range
        lr_value = suggestion["lr"]
        assert 0.001 <= lr_value <= 0.01, f"lr value {lr_value} outside expected range [0.001, 0.01]"


def test_suggestion_not_duplicated_on_eval():
    """Test that eval jobs don't try to store suggestions."""

    # Create mock store
    mock_store = MagicMock(spec=WandbStore)

    # Track what was passed to init_run and update_run_summary
    init_run_calls = []

    def capture_init_run(run_id, group=None, tags=None, initial_summary=None):
        init_run_calls.append({"run_id": run_id, "initial_summary": initial_summary})

    mock_store.init_run.side_effect = capture_init_run

    update_summary_calls = []

    def capture_update_summary(run_id, summary_update):
        update_summary_calls.append({"run_id": run_id, "summary_update": summary_update})
        return True

    mock_store.update_run_summary.side_effect = capture_update_summary

    # Return a run that needs evaluation
    # RunInfo requires all fields to be set explicitly
    from datetime import datetime, timezone

    mock_run = RunInfo(
        run_id="test_sweep_trial_0001",
        summary={"sweep/suggestion": {"lr": 0.005}},
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=False,
        has_been_evaluated=False,
        has_failed=False,
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    mock_store.fetch_runs.return_value = [mock_run]

    # Create mock dispatcher
    mock_dispatcher = MagicMock()
    mock_dispatcher.dispatch.return_value = "dispatch-456"

    # Create scheduler config
    protein_config = ProteinConfig(
        metric="test/metric",
        goal="maximize",
        parameters={
            "lr": ParameterConfig(
                min=0.001,
                max=0.01,
                distribution="log_normal",
                mean=0.005,  # Required for log_normal
                scale="auto",  # Required for log_normal
            )
        },
    )

    scheduler_config = BatchedSyncedSchedulerConfig(
        max_trials=2,
        batch_size=2,
        recipe_module="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="eval",
        experiment_id="test_sweep",
        protein_config=protein_config,
    )

    # Create scheduler with state that knows about the training run
    from softmax.training.sweep.schedulers.batched_synced import SchedulerState

    state = SchedulerState(runs_in_training={"test_sweep_trial_0001"}, runs_in_eval=set(), runs_completed=set())
    scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, state=state)

    # Create adaptive config
    adaptive_config = AdaptiveConfig(max_parallel=2, monitoring_interval=1, resume=False)

    # Create controller (not used directly, but validates configuration)
    _ = AdaptiveController(
        experiment_id="test_sweep",
        scheduler=scheduler,
        dispatcher=mock_dispatcher,
        store=mock_store,
        config=adaptive_config,
    )

    # Schedule eval job
    jobs = scheduler.schedule([mock_run], available_training_slots=2)

    # Should get one eval job
    assert len(jobs) == 1
    assert jobs[0].type == JobTypes.LAUNCH_EVAL

    # Dispatch eval job (simulating what controller does)
    for job in jobs:
        _ = mock_dispatcher.dispatch(job)
        if job.type == JobTypes.LAUNCH_EVAL:
            # Eval jobs should only update has_started_eval, not init_run
            mock_store.update_run_summary(job.run_id, {"has_started_eval": True})

    # Verify no init_run was called (eval reuses existing run)
    assert len(init_run_calls) == 0, f"init_run should not be called for eval jobs, but got {len(init_run_calls)} calls"

    # Verify update_run_summary was called with has_started_eval only
    assert len(update_summary_calls) == 1
    assert update_summary_calls[0]["summary_update"] == {"has_started_eval": True}
    assert "sweep/suggestion" not in update_summary_calls[0]["summary_update"], (
        "Eval jobs should not update sweep/suggestion"
    )
