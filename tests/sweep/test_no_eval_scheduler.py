"""Tests for NoEvalSweepScheduler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.sweep.schedulers.top_policy_no_eval import (
    BatchedSyncedSchedulerConfig,
    NoEvalSweepScheduler,
    SchedulerState,
)


@pytest.fixture
def mock_store():
    """Create a mock store."""
    store = MagicMock()
    store.update_run_summary = MagicMock()
    return store


@pytest.fixture
def mock_protein_config():
    """Create a mock protein config."""
    config = MagicMock()
    config.metric = "accuracy"
    return config


@pytest.fixture
def scheduler_config(mock_protein_config):
    """Create a test scheduler config."""
    return BatchedSyncedSchedulerConfig(
        max_trials=10,
        batch_size=4,
        experiment_id="test_exp",
        recipe_module="test.recipe",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        gpus=1,
        nodes=1,
        protein_config=mock_protein_config,
    )


@pytest.fixture
def scheduler(scheduler_config, mock_store):
    """Create a scheduler instance."""
    with patch("metta.sweep.schedulers.top_policy_no_eval.ProteinOptimizer"):
        return NoEvalSweepScheduler(scheduler_config, mock_store)


class TestSchedulerState:
    """Tests for SchedulerState."""

    def test_model_dump(self):
        """Test serialization of state."""
        state = SchedulerState(
            runs_in_training={"run1", "run2"},
            runs_completed={"run3"},
            top_score_per_run={"run3": 0.95},
        )
        dumped = state.model_dump()
        assert set(dumped["runs_in_training"]) == {"run1", "run2"}
        assert set(dumped["runs_completed"]) == {"run3"}
        # top_score_per_run is not serialized in model_dump

    def test_model_validate(self):
        """Test deserialization of state."""
        data = {
            "runs_in_training": ["run1", "run2"],
            "runs_completed": ["run3"],
        }
        state = SchedulerState.model_validate(data)
        assert state.runs_in_training == {"run1", "run2"}
        assert state.runs_completed == {"run3"}
        assert state.top_score_per_run == {}


class TestNoEvalSweepScheduler:
    """Tests for NoEvalSweepScheduler."""

    def test_initialization(self, scheduler_config, mock_store):
        """Test scheduler initialization."""
        with patch("metta.sweep.schedulers.top_policy_no_eval.ProteinOptimizer") as MockOptimizer:
            scheduler = NoEvalSweepScheduler(scheduler_config, mock_store)
            assert scheduler.config == scheduler_config
            assert scheduler.store == mock_store
            assert isinstance(scheduler.state, SchedulerState)
            MockOptimizer.assert_called_once_with(scheduler_config.protein_config)

    def test_update_state_empty_runs(self, scheduler):
        """Test state update with empty runs list."""
        scheduler.state.runs_in_training.add("run1")
        scheduler._update_state_from_runs([])
        # State should remain unchanged
        assert "run1" in scheduler.state.runs_in_training

    def test_update_state_training_done(self, scheduler, mock_store):
        """Test state update when training completes."""
        scheduler.state.runs_in_training.add("run1")
        scheduler.state.top_score_per_run["run1"] = 0.8

        runs = [
            RunInfo(
                run_id="run1",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"accuracy": 0.9},
                cost=100.0,
            )
        ]

        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_training
        assert "run1" in scheduler.state.runs_completed
        mock_store.update_run_summary.assert_called_once_with(
            "run1", {"sweep/score": 0.9, "sweep/cost": 100.0}
        )

    def test_update_state_failed_run(self, scheduler):
        """Test state update when run fails."""
        scheduler.state.runs_in_training.add("run1")

        runs = [
            RunInfo(
                run_id="run1",
                has_failed=True,
                summary={},
                cost=50.0,
            )
        ]

        scheduler._update_state_from_runs(runs)

        assert "run1" not in scheduler.state.runs_in_training
        assert "run1" not in scheduler.state.runs_completed

    def test_track_top_scores(self, scheduler):
        """Test tracking of top scores across runs."""
        runs = [
            RunInfo(
                run_id="run1",
                has_started_training=True,
                summary={"accuracy": 0.7},
                cost=0,
            ),
            RunInfo(
                run_id="run1",
                has_started_training=True,
                summary={"accuracy": 0.9},
                cost=0,
            ),
            RunInfo(
                run_id="run2",
                has_started_training=True,
                summary={"accuracy": 0.8},
                cost=0,
            ),
        ]

        scheduler._update_state_from_runs(runs[:1])
        assert scheduler.state.top_score_per_run["run1"] == 0.7

        scheduler._update_state_from_runs(runs[1:2])
        assert scheduler.state.top_score_per_run["run1"] == 0.9  # Should keep max

        scheduler._update_state_from_runs(runs[2:])
        assert scheduler.state.top_score_per_run["run2"] == 0.8

    def test_schedule_wait_for_training(self, scheduler):
        """Test scheduler waits for training to complete."""
        scheduler.state.runs_in_training.add("run1")

        jobs = scheduler.schedule([], available_training_slots=4)

        assert len(jobs) == 0  # Should wait

    def test_schedule_max_trials_reached(self, scheduler):
        """Test scheduler stops at max trials."""
        runs = [RunInfo(run_id=f"run{i}",
                       has_started_training=True,
                       has_completed_training=True,
                       has_started_eval=False)
                for i in range(10)]

        jobs = scheduler.schedule(runs, available_training_slots=4)

        assert len(jobs) == 0  # Max trials reached

    def test_schedule_new_batch(self, scheduler):
        """Test scheduling a new batch of training jobs."""
        runs = []
        scheduler.optimizer.suggest = MagicMock(
            return_value=[
                {"lr": 0.001, "batch_size": 32},
                {"lr": 0.01, "batch_size": 64},
            ]
        )

        jobs = scheduler.schedule(runs, available_training_slots=4)

        assert len(jobs) == 2
        assert all(isinstance(job, JobDefinition) for job in jobs)
        # Check that 2 runs were added to training
        assert len(scheduler.state.runs_in_training) == 2
        # Run IDs should start with experiment_id and include trial numbers
        run_ids = list(scheduler.state.runs_in_training)
        assert all(run_id.startswith("test_exp_trial_") for run_id in run_ids)
        assert jobs[0].metadata["sweep/suggestion"] == {"lr": 0.001, "batch_size": 32}
        assert jobs[1].metadata["sweep/suggestion"] == {"lr": 0.01, "batch_size": 64}

    def test_schedule_limited_by_capacity(self, scheduler):
        """Test scheduling limited by available slots."""
        scheduler.config.batch_size = 10  # Want 10 but only 2 slots
        scheduler.optimizer.suggest = MagicMock(
            return_value=[{"lr": 0.001}, {"lr": 0.01}]
        )

        jobs = scheduler.schedule([], available_training_slots=2)

        assert len(jobs) == 2  # Limited by capacity
        scheduler.optimizer.suggest.assert_called_once_with([], n_suggestions=2)

    def test_is_experiment_complete(self, scheduler):
        """Test experiment completion detection."""
        # Not complete with fewer runs
        runs = [RunInfo(run_id=f"run{i}",
                       has_started_training=True,
                       has_completed_training=True,
                       has_started_eval=False)
                for i in range(5)]
        assert not scheduler.is_experiment_complete(runs)

        # Complete with max trials reached
        runs = [RunInfo(run_id=f"run{i}",
                       has_started_training=True,
                       has_completed_training=True,
                       has_started_eval=False)
                for i in range(10)]
        assert scheduler.is_experiment_complete(runs)

        # Also counts failed runs
        runs = [RunInfo(run_id=f"run{i}",
                       has_started_training=True,
                       has_completed_training=(i < 8),
                       has_started_eval=False,
                       has_failed=(i >= 8))
                for i in range(10)]
        assert scheduler.is_experiment_complete(runs)

    def test_collect_observations(self, scheduler):
        """Test observation collection from completed runs."""
        runs = [
            RunInfo(
                run_id="run1",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={
                    "sweep/score": 0.9,
                    "sweep/cost": 100,
                    "sweep/suggestion": {"lr": 0.001},
                },
            ),
            RunInfo(
                run_id="run2",
                has_started_training=True,
                has_completed_training=False,  # Still running, should be skipped
                summary={"sweep/score": 0.8},
            ),
            RunInfo(
                run_id="run3",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"sweep/score": 0.85, "sweep/cost": 150},
            ),
        ]

        observations = scheduler._collect_observations(runs)

        assert len(observations) == 2
        assert observations[0] == {
            "score": 0.9,
            "cost": 100.0,
            "suggestion": {"lr": 0.001},
        }
        assert observations[1] == {
            "score": 0.85,
            "cost": 150.0,
            "suggestion": {},
        }

    def test_collect_observations_malformed(self, scheduler):
        """Test observation collection handles malformed data."""
        runs = [
            RunInfo(
                run_id="run1",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"sweep/score": "not_a_number"},  # Invalid
            ),
            RunInfo(
                run_id="run2",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"sweep/score": 0.9},  # Valid
            ),
        ]

        observations = scheduler._collect_observations(runs)

        assert len(observations) == 1  # Only valid one
        assert observations[0]["score"] == 0.9

    def test_full_scheduling_cycle(self, scheduler, mock_store):
        """Test a full scheduling cycle from start to completion."""
        # Initial state: no runs
        scheduler.optimizer.suggest = MagicMock(
            return_value=[{"lr": 0.001}, {"lr": 0.01}]
        )

        # Schedule first batch
        jobs = scheduler.schedule([], available_training_slots=4)
        assert len(jobs) == 2
        assert len(scheduler.state.runs_in_training) == 2

        # Get the actual run IDs that were generated
        generated_run_ids = [job.run_id for job in jobs]
        assert all(run_id.startswith("test_exp_trial_") for run_id in generated_run_ids)

        # Simulate runs in progress using the actual run IDs
        runs = [
            RunInfo(run_id=generated_run_ids[0], has_started_training=True, has_completed_training=False),
            RunInfo(run_id=generated_run_ids[1], has_started_training=True, has_completed_training=False),
        ]

        # Should wait for training to complete
        jobs = scheduler.schedule(runs, available_training_slots=4)
        assert len(jobs) == 0

        # Simulate training completion with the actual run IDs
        runs = [
            RunInfo(
                run_id=generated_run_ids[0],
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"accuracy": 0.9, "sweep/suggestion": {"lr": 0.001}},
                cost=100,
            ),
            RunInfo(
                run_id=generated_run_ids[1],
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
                summary={"accuracy": 0.85, "sweep/suggestion": {"lr": 0.01}},
                cost=110,
            ),
        ]

        # Update state and schedule next batch
        scheduler.optimizer.suggest = MagicMock(
            return_value=[{"lr": 0.005}]
        )

        jobs = scheduler.schedule(runs, available_training_slots=4)

        # Should have scheduled new batch after previous completed
        assert len(jobs) == 1
        # Check that a new run was added to training
        assert len(scheduler.state.runs_in_training) == 1
        new_run_id = list(scheduler.state.runs_in_training)[0]
        assert new_run_id.startswith("test_exp_trial_0003")
        assert len(scheduler.state.runs_completed) == 2

        # Verify store updates were called
        assert mock_store.update_run_summary.call_count == 2