"""Unit tests for BatchedSyncedOptimizingScheduler state tracking.

Tests all edge cases of internal state management including:
- State transitions between training, eval, and completed
- Empty runs list handling
- Duplicate dispatch prevention
- Batch synchronization constraints
- Failure/stale handling
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from metta.adaptive.models import JobStatus, RunInfo
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
    SchedulerState,
)


@pytest.fixture
def protein_config():
    """Create a simple protein config for testing."""
    return ProteinConfig(
        metric="test/score",
        goal="maximize",
        parameters={"lr": ParameterConfig(min=1e-5, max=1e-3, distribution="log_normal", mean=1e-4, scale="auto")},
    )


@pytest.fixture
def scheduler_config(protein_config):
    """Create scheduler config for testing."""
    return BatchedSyncedSchedulerConfig(
        max_trials=10,
        batch_size=3,
        experiment_id="test_exp",
        recipe_module="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        protein_config=protein_config,
    )


@pytest.fixture
def scheduler(scheduler_config):
    """Create scheduler instance with mocked optimizer."""
    with patch("metta.sweep.optimizer.protein.ProteinOptimizer") as mock_optimizer_class:
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config)
        scheduler.optimizer = mock_optimizer
        return scheduler


def create_run(run_id: str, status: JobStatus, summary: dict = None) -> RunInfo:
    """Helper to create RunInfo objects for testing.

    Note: STALE status is computed based on last_updated_at being >20 minutes old,
    so we handle it specially.
    """
    from datetime import timedelta

    # For STALE, set last_updated_at to be old
    if status == JobStatus.STALE:
        last_updated_at = datetime.now(timezone.utc) - timedelta(minutes=25)
        has_started_training = True
        has_completed_training = False
        has_failed = False
    else:
        last_updated_at = datetime.now(timezone.utc)
        has_started_training = status != JobStatus.PENDING
        has_completed_training = status in [JobStatus.TRAINING_DONE_NO_EVAL, JobStatus.IN_EVAL, JobStatus.COMPLETED]
        has_failed = status == JobStatus.FAILED

    return RunInfo(
        run_id=run_id,
        group="test_exp",
        created_at=datetime.now(timezone.utc),
        last_updated_at=last_updated_at,
        summary=summary or {},
        has_started_training=has_started_training,
        has_completed_training=has_completed_training,
        has_started_eval=(status in [JobStatus.IN_EVAL, JobStatus.COMPLETED]),
        has_been_evaluated=(status == JobStatus.COMPLETED),
        has_failed=has_failed,
        cost=100.0,
        runtime=3600.0,
    )


class TestSchedulerStateManagement:
    """Test SchedulerState class functionality."""

    def test_state_initialization(self):
        """Test that state initializes with empty sets."""
        state = SchedulerState()
        assert state.runs_in_training == set()
        assert state.runs_in_eval == set()
        assert state.runs_completed == set()

    def test_state_serialization(self):
        """Test state serialization to dict."""
        state = SchedulerState(
            runs_in_training={"run1", "run2"}, runs_in_eval={"run3"}, runs_completed={"run4", "run5"}
        )

        dumped = state.model_dump()
        assert set(dumped["runs_in_training"]) == {"run1", "run2"}
        assert set(dumped["runs_in_eval"]) == {"run3"}
        assert set(dumped["runs_completed"]) == {"run4", "run5"}

    def test_state_deserialization(self):
        """Test state deserialization from dict."""
        data = {"runs_in_training": ["run1", "run2"], "runs_in_eval": ["run3"], "runs_completed": ["run4", "run5"]}

        state = SchedulerState.model_validate(data)
        assert state.runs_in_training == {"run1", "run2"}
        assert state.runs_in_eval == {"run3"}
        assert state.runs_completed == {"run4", "run5"}

    def test_state_deserialization_with_missing_fields(self):
        """Test state deserialization handles missing fields gracefully."""
        state = SchedulerState.model_validate({})
        assert state.runs_in_training == set()
        assert state.runs_in_eval == set()
        assert state.runs_completed == set()


class TestStateUpdateLogic:
    """Test the _update_state_from_runs method."""

    def test_empty_runs_with_empty_state(self, scheduler):
        """Test that empty runs list with empty state causes no issues."""
        scheduler._update_state_from_runs([])
        assert scheduler.state.runs_in_training == set()
        assert scheduler.state.runs_in_eval == set()
        assert scheduler.state.runs_completed == set()

    def test_empty_runs_with_active_state_logs_warning(self, scheduler, caplog):
        """Test that empty runs with active state logs a warning."""
        scheduler.state.runs_in_training = {"run1", "run2"}
        scheduler.state.runs_in_eval = {"run3"}

        scheduler._update_state_from_runs([])

        assert "WARNING: Received empty runs list" in caplog.text
        assert "2 runs in training" in caplog.text
        assert "1 runs in eval" in caplog.text
        # State should not be modified
        assert scheduler.state.runs_in_training == {"run1", "run2"}
        assert scheduler.state.runs_in_eval == {"run3"}

    def test_training_to_training_done_transition(self, scheduler):
        """Test run transitioning from training to training done."""
        scheduler.state.runs_in_training = {"run1"}

        runs = [create_run("run1", JobStatus.TRAINING_DONE_NO_EVAL)]
        scheduler._update_state_from_runs(runs)

        # Should remain in runs_in_training until eval is dispatched
        assert "run1" in scheduler.state.runs_in_training
        assert "run1" not in scheduler.state.runs_in_eval

    def test_training_to_failed_transition(self, scheduler):
        """Test run transitioning from training to failed."""
        scheduler.state.runs_in_training = {"run1", "run2"}

        runs = [create_run("run1", JobStatus.FAILED), create_run("run2", JobStatus.IN_TRAINING)]
        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_training
        assert "run2" in scheduler.state.runs_in_training
        assert scheduler.state.runs_in_eval == set()

    def test_training_to_stale_transition(self, scheduler):
        """Test run transitioning from training to stale."""
        scheduler.state.runs_in_training = {"run1"}

        runs = [create_run("run1", JobStatus.STALE)]
        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_training

    def test_eval_to_completed_transition(self, scheduler):
        """Test run transitioning from eval to completed."""
        scheduler.state.runs_in_eval = {"run1", "run2"}

        runs = [create_run("run1", JobStatus.COMPLETED), create_run("run2", JobStatus.IN_EVAL)]
        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_eval
        assert "run1" in scheduler.state.runs_completed
        assert "run2" in scheduler.state.runs_in_eval
        assert "run2" not in scheduler.state.runs_completed

    def test_eval_to_failed_transition(self, scheduler):
        """Test eval failure moves run to completed (no retry)."""
        scheduler.state.runs_in_eval = {"run1"}

        runs = [create_run("run1", JobStatus.FAILED)]
        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_eval
        assert "run1" in scheduler.state.runs_completed

    def test_runs_not_in_state_are_ignored(self, scheduler):
        """Test that runs not tracked in state don't cause issues."""
        scheduler.state.runs_in_training = {"run1"}

        runs = [
            create_run("run1", JobStatus.IN_TRAINING),
            create_run("run2", JobStatus.COMPLETED),  # Not tracked
            create_run("run3", JobStatus.FAILED),  # Not tracked
        ]
        scheduler._update_state_from_runs(runs)

        # Only run1 should be in state
        assert scheduler.state.runs_in_training == {"run1"}
        assert scheduler.state.runs_in_eval == set()
        assert scheduler.state.runs_completed == set()


class TestEvalScheduling:
    """Test evaluation job scheduling with state tracking."""

    def test_schedule_eval_for_training_done(self, scheduler):
        """Test that eval is scheduled for training-done runs."""
        runs = [create_run("run1", JobStatus.TRAINING_DONE_NO_EVAL)]

        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 1
        assert "evaluate" in jobs[0].cmd
        assert jobs[0].run_id == "run1"
        # State should be updated
        assert "run1" not in scheduler.state.runs_in_training
        assert "run1" in scheduler.state.runs_in_eval

    def test_no_duplicate_eval_dispatch(self, scheduler):
        """Test that eval is not dispatched twice for the same run."""
        scheduler.state.runs_in_eval = {"run1"}
        runs = [create_run("run1", JobStatus.TRAINING_DONE_NO_EVAL)]

        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0  # No job should be created
        assert "run1" in scheduler.state.runs_in_eval

    def test_multiple_evals_scheduled_together(self, scheduler):
        """Test multiple eval jobs can be scheduled in one call."""
        scheduler.state.runs_in_training = {"run1", "run2", "run3"}
        runs = [
            create_run("run1", JobStatus.TRAINING_DONE_NO_EVAL),
            create_run("run2", JobStatus.TRAINING_DONE_NO_EVAL),
            create_run("run3", JobStatus.IN_TRAINING),
        ]

        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 2
        assert all("evaluate" in job.cmd for job in jobs)
        assert {job.run_id for job in jobs} == {"run1", "run2"}
        assert scheduler.state.runs_in_training == {"run3"}
        assert scheduler.state.runs_in_eval == {"run1", "run2"}


class TestBatchSynchronization:
    """Test batch synchronization constraints."""

    def test_no_new_training_while_training_in_progress(self, scheduler):
        """Test that new training batch is not started while training is in progress."""
        scheduler.state.runs_in_training = {"run1"}
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        runs = [create_run("run1", JobStatus.IN_TRAINING)]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0
        assert scheduler.optimizer.suggest.call_count == 0

    def test_no_new_training_while_eval_in_progress(self, scheduler):
        """Test that new training batch is not started while eval is in progress."""
        scheduler.state.runs_in_eval = {"run1"}
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        runs = [create_run("run1", JobStatus.IN_EVAL)]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0
        assert scheduler.optimizer.suggest.call_count == 0

    def test_new_batch_starts_when_all_complete(self, scheduler):
        """Test that new batch starts only when all runs are complete."""
        scheduler.state.runs_completed = {"run1", "run2"}
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}, {"lr": 0.002}, {"lr": 0.003}]

        runs = [
            create_run("run1", JobStatus.COMPLETED, {"sweep/score": 0.8, "sweep/suggestion": {"lr": 0.0005}}),
            create_run("run2", JobStatus.COMPLETED, {"sweep/score": 0.9, "sweep/suggestion": {"lr": 0.0007}}),
        ]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 3  # Full batch
        assert all("train" in job.cmd for job in jobs)
        assert len(scheduler.state.runs_in_training) == 3
        assert scheduler.optimizer.suggest.call_count == 1

    def test_batch_size_respects_available_slots(self, scheduler):
        """Test that batch size respects available training slots."""
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        jobs = scheduler.schedule([], available_training_slots=1)

        assert len(jobs) == 1
        # Should only request 1 suggestion due to slot limit
        scheduler.optimizer.suggest.assert_called_with([], n_suggestions=1)

    def test_batch_size_respects_remaining_trials(self, scheduler):
        """Test that batch size respects remaining trials to max_trials."""
        scheduler.config.max_trials = 5
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        # Already have 4 runs, only 1 remaining
        runs = [create_run(f"run{i}", JobStatus.COMPLETED) for i in range(4)]
        jobs = scheduler.schedule(runs, available_training_slots=10)

        assert len(jobs) == 1
        scheduler.optimizer.suggest.assert_called_with([], n_suggestions=1)


class TestMaxTrialsHandling:
    """Test max_trials limit handling."""

    def test_no_new_jobs_when_max_trials_reached(self, scheduler):
        """Test that no new training jobs are created when max_trials is reached."""
        scheduler.config.max_trials = 3
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        runs = [
            create_run("run1", JobStatus.COMPLETED),
            create_run("run2", JobStatus.COMPLETED),
            create_run("run3", JobStatus.IN_TRAINING),
        ]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0
        assert scheduler.optimizer.suggest.call_count == 0

    def test_eval_still_scheduled_at_max_trials(self, scheduler):
        """Test that eval jobs are still scheduled even at max_trials."""
        scheduler.config.max_trials = 2

        runs = [create_run("run1", JobStatus.COMPLETED), create_run("run2", JobStatus.TRAINING_DONE_NO_EVAL)]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 1
        assert "evaluate" in jobs[0].cmd

    def test_is_experiment_complete(self, scheduler):
        """Test is_experiment_complete logic."""
        scheduler.config.max_trials = 3

        # Not complete - only 2 completed
        runs = [
            create_run("run1", JobStatus.COMPLETED),
            create_run("run2", JobStatus.COMPLETED),
            create_run("run3", JobStatus.IN_TRAINING),
        ]
        assert not scheduler.is_experiment_complete(runs)

        # Complete - 3 completed
        runs[2] = create_run("run3", JobStatus.COMPLETED)
        assert scheduler.is_experiment_complete(runs)


class TestObservationCollection:
    """Test observation collection for optimizer."""

    def test_collect_observations_from_completed_runs(self, scheduler):
        """Test that observations are correctly collected from completed runs."""
        runs = [
            create_run(
                "run1",
                JobStatus.COMPLETED,
                {"sweep/score": 0.85, "sweep/cost": 150.0, "sweep/suggestion": {"lr": 0.001}},
            ),
            create_run(
                "run2",
                JobStatus.COMPLETED,
                {"sweep/score": 0.90, "sweep/cost": 200.0, "sweep/suggestion": {"lr": 0.002}},
            ),
            create_run("run3", JobStatus.IN_TRAINING),  # Should be ignored
        ]

        obs = scheduler._collect_observations(runs)

        assert len(obs) == 2
        assert obs[0]["score"] == 0.85
        assert obs[0]["cost"] == 150.0
        assert obs[0]["suggestion"]["lr"] == 0.001
        assert obs[1]["score"] == 0.90

    def test_skip_runs_without_scores(self, scheduler):
        """Test that runs without scores are skipped."""
        runs = [
            create_run("run1", JobStatus.COMPLETED, {"sweep/suggestion": {"lr": 0.001}}),
            create_run("run2", JobStatus.COMPLETED, {"sweep/score": 0.9, "sweep/suggestion": {"lr": 0.002}}),
        ]

        obs = scheduler._collect_observations(runs)

        assert len(obs) == 1
        assert obs[0]["score"] == 0.9

    def test_handle_malformed_observations(self, scheduler):
        """Test that malformed observations are skipped gracefully."""
        runs = [
            create_run(
                "run1",
                JobStatus.COMPLETED,
                {
                    "sweep/score": "not_a_number",  # Invalid
                    "sweep/suggestion": {"lr": 0.001},
                },
            ),
            create_run("run2", JobStatus.COMPLETED, {"sweep/score": 0.9, "sweep/suggestion": {"lr": 0.002}}),
        ]

        obs = scheduler._collect_observations(runs)

        assert len(obs) == 1
        assert obs[0]["score"] == 0.9


class TestCompleteWorkflow:
    """Test complete workflow scenarios."""

    def test_full_batch_lifecycle(self, scheduler):
        """Test a complete batch lifecycle from training to completion."""
        # Start with empty state
        assert scheduler.state.runs_in_training == set()

        # 1. Schedule first batch of training
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}, {"lr": 0.002}, {"lr": 0.003}]
        jobs = scheduler.schedule([], available_training_slots=5)

        assert len(jobs) == 3
        assert all("train" in job.cmd for job in jobs)
        assert len(scheduler.state.runs_in_training) == 3
        run_ids = {job.run_id for job in jobs}

        # 2. Training in progress - no new jobs
        runs = [create_run(rid, JobStatus.IN_TRAINING) for rid in run_ids]
        jobs = scheduler.schedule(runs, available_training_slots=5)
        assert len(jobs) == 0

        # 3. Training done - schedule evals
        runs = [create_run(rid, JobStatus.TRAINING_DONE_NO_EVAL) for rid in run_ids]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 3
        assert all("evaluate" in job.cmd for job in jobs)
        assert scheduler.state.runs_in_training == set()
        assert scheduler.state.runs_in_eval == run_ids

        # 4. Eval in progress - no new jobs
        runs = [create_run(rid, JobStatus.IN_EVAL) for rid in run_ids]
        jobs = scheduler.schedule(runs, available_training_slots=5)
        assert len(jobs) == 0

        # 5. All completed - ready for next batch
        runs = [
            create_run(
                rid, JobStatus.COMPLETED, {"sweep/score": 0.8 + i * 0.05, "sweep/suggestion": {"lr": 0.001 + i * 0.001}}
            )
            for i, rid in enumerate(run_ids)
        ]
        scheduler.optimizer.suggest.return_value = [{"lr": 0.004}]
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 1  # Next batch starts
        assert scheduler.state.runs_completed == run_ids

    def test_mixed_status_handling(self, scheduler):
        """Test handling of runs in different states simultaneously."""
        scheduler.state.runs_in_training = {"run1", "run2"}
        scheduler.state.runs_in_eval = {"run3"}
        scheduler.state.runs_completed = {"run4"}

        runs = [
            create_run("run1", JobStatus.TRAINING_DONE_NO_EVAL),  # Ready for eval
            create_run("run2", JobStatus.FAILED),  # Failed during training
            create_run("run3", JobStatus.COMPLETED, {"sweep/score": 0.9}),  # Just completed eval
            create_run("run4", JobStatus.COMPLETED, {"sweep/score": 0.85}),  # Already completed
        ]

        jobs = scheduler.schedule(runs, available_training_slots=5)

        # Should schedule eval for run1
        assert len(jobs) == 1
        assert jobs[0].run_id == "run1"
        assert "evaluate" in jobs[0].cmd

        # Check final state
        assert scheduler.state.runs_in_training == set()
        assert scheduler.state.runs_in_eval == {"run1"}
        assert scheduler.state.runs_completed == {"run3", "run4"}

    def test_recovery_from_stale_runs(self, scheduler):
        """Test that scheduler recovers from stale runs and continues."""
        # Start with run1 in training, run2 in eval (more realistic scenario)
        scheduler.state.runs_in_training = {"run1"}
        scheduler.state.runs_in_eval = {"run2"}
        scheduler.optimizer.suggest.return_value = [{"lr": 0.005}]

        # One run went stale, one completed eval
        runs = [
            create_run("run1", JobStatus.STALE),
            create_run("run2", JobStatus.COMPLETED, {"sweep/score": 0.7, "sweep/suggestion": {"lr": 0.001}}),
        ]

        jobs = scheduler.schedule(runs, available_training_slots=5)

        # Should start new batch since no runs are active
        assert len(jobs) == 1
        assert "train" in jobs[0].cmd
        assert scheduler.state.runs_in_training == {jobs[0].run_id}
        assert scheduler.state.runs_in_eval == set()
        assert scheduler.state.runs_completed == {"run2"}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_available_slots(self, scheduler):
        """Test behavior when no training slots are available."""
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]

        jobs = scheduler.schedule([], available_training_slots=0)

        assert len(jobs) == 0
        assert scheduler.optimizer.suggest.call_count == 0

    def test_optimizer_returns_no_suggestions(self, scheduler):
        """Test handling when optimizer returns no suggestions."""
        scheduler.optimizer.suggest.return_value = []

        jobs = scheduler.schedule([], available_training_slots=5)

        assert len(jobs) == 0
        assert len(scheduler.state.runs_in_training) == 0

    def test_optimizer_returns_fewer_suggestions(self, scheduler):
        """Test when optimizer returns fewer suggestions than requested."""
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]  # Only 1

        jobs = scheduler.schedule([], available_training_slots=5)

        assert len(jobs) == 1  # Should create job for the one suggestion
        scheduler.optimizer.suggest.assert_called_with([], n_suggestions=3)  # Requested 3

    def test_state_persistence_between_calls(self, scheduler):
        """Test that state persists correctly between schedule calls."""
        # First call - start training
        scheduler.optimizer.suggest.return_value = [{"lr": 0.001}]
        jobs1 = scheduler.schedule([], available_training_slots=5)
        run_id = jobs1[0].run_id

        assert scheduler.state.runs_in_training == {run_id}

        # Second call - training still in progress
        runs = [create_run(run_id, JobStatus.IN_TRAINING)]
        jobs2 = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs2) == 0
        assert scheduler.state.runs_in_training == {run_id}

        # Third call - training done, schedule eval
        runs = [create_run(run_id, JobStatus.TRAINING_DONE_NO_EVAL)]
        jobs3 = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs3) == 1
        assert scheduler.state.runs_in_eval == {run_id}
        assert scheduler.state.runs_in_training == set()
