"""Tests for TrainAndEvalScheduler."""

import pytest

from metta.adaptive.models import JobTypes, RunInfo
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler


class TestTrainAndEvalScheduler:
    """Test the train-and-eval scheduler logic."""

    @pytest.fixture
    def config(self):
        """Basic scheduler configuration."""
        return TrainAndEvalConfig(
            max_trials=3,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            gpus=1,
        )

    @pytest.fixture
    def scheduler(self, config):
        """Basic scheduler instance."""
        return TrainAndEvalScheduler(config)

    def test_initialization(self, scheduler, config):
        """Test scheduler initialization."""
        assert scheduler.config is config
        assert scheduler.state is None

    def test_is_experiment_complete_max_trials(self, scheduler):
        """Test experiment completion when max_trials reached."""
        from datetime import datetime, timezone

        # Create runs up to max_trials with COMPLETED status
        runs = [
            RunInfo(
                run_id=f"trial_{i:04d}",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
                summary={"sweep/score": 0.5, "sweep/cost": 1.0, "sweep/suggestion": {}},
            )
            for i in range(scheduler.config.max_trials)
        ]

        assert scheduler.is_experiment_complete(runs) is True

    def test_is_experiment_complete_not_done(self, scheduler):
        """Test experiment not complete when under max_trials."""
        runs = [RunInfo(run_id="trial_0001", has_started_training=True)]
        assert scheduler.is_experiment_complete(runs) is False

    def test_is_experiment_complete_empty_runs(self, scheduler):
        """Test experiment not complete with no runs."""
        assert scheduler.is_experiment_complete([]) is False

    def test_schedule_first_training_job(self, scheduler):
        """Test creating training jobs when no runs exist."""
        jobs = scheduler.schedule(runs=[], available_training_slots=2)

        # Should create 2 jobs (up to available slots)
        assert len(jobs) == 2

        for job in jobs:
            assert job.type == JobTypes.LAUNCH_TRAINING
            assert "trial_" in job.run_id  # Uses experiment_id pattern
            assert job.cmd == "experiments.recipes.arena.train"
            assert job.gpus == 1
            assert job.nodes == 1

    def test_schedule_respects_available_slots(self, scheduler):
        """Test scheduler respects available training slot limits."""
        jobs = scheduler.schedule(runs=[], available_training_slots=0)

        # No training jobs should be created when no slots available
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
        assert len(training_jobs) == 0

    def test_schedule_multiple_training_jobs(self, scheduler):
        """Test creating multiple training jobs when slots available."""
        scheduler.config.max_trials = 10  # High limit

        jobs = scheduler.schedule(runs=[], available_training_slots=3)

        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
        assert len(training_jobs) == 3

        # All jobs should have unique run_ids
        run_ids = [job.run_id for job in training_jobs]
        assert len(set(run_ids)) == 3

    def test_schedule_evaluation_jobs(self, scheduler):
        """Test scheduling evaluation for completed training runs."""
        # Setup: one run with training complete but no eval started
        completed_run = RunInfo(
            run_id="trial_0001", has_started_training=True, has_completed_training=True, has_started_eval=False
        )
        runs = [completed_run]

        jobs = scheduler.schedule(runs=runs, available_training_slots=1)

        # Should create both eval job and new training job
        eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]

        assert len(eval_jobs) == 1
        assert len(training_jobs) == 1

        eval_job = eval_jobs[0]
        assert eval_job.run_id == "trial_0001"
        assert eval_job.cmd == "experiments.recipes.arena.evaluate"

    def test_schedule_no_duplicate_evals(self, scheduler):
        """Test that evaluation jobs are not created for runs already in eval."""
        # Setup: run with eval already started
        eval_in_progress = RunInfo(
            run_id="trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=False,
        )
        runs = [eval_in_progress]

        jobs = scheduler.schedule(runs=runs, available_training_slots=1)

        # Should only create training job, no eval job
        eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]

        assert len(eval_jobs) == 0
        assert len(training_jobs) == 1

    def test_schedule_stops_at_max_trials(self, scheduler):
        """Test that no new training jobs created when max_trials reached."""
        # Create max_trials runs (any status is fine, just count)
        runs = [
            RunInfo(
                run_id=f"trial_{i:04d}", has_started_training=True, has_completed_training=True, has_been_evaluated=True
            )
            for i in range(scheduler.config.max_trials)
        ]

        jobs = scheduler.schedule(runs=runs, available_training_slots=5)

        # Should create no new training jobs
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]
        assert len(training_jobs) == 0

    def test_schedule_with_overrides(self, config):
        """Test that scheduler config overrides are applied to jobs."""
        config.train_overrides = {"learning_rate": 0.001}
        scheduler = TrainAndEvalScheduler(config)

        jobs = scheduler.schedule(runs=[], available_training_slots=1)

        training_job = jobs[0]
        assert training_job.overrides == {"learning_rate": 0.001}

    def test_run_id_generation_pattern(self, scheduler):
        """Test that run IDs follow expected pattern."""
        scheduler.config.experiment_id = "my_experiment"

        jobs = scheduler.schedule(runs=[], available_training_slots=1)

        run_id = jobs[0].run_id
        assert run_id.startswith("my_experiment_trial_")
        assert len(run_id) > len("my_experiment_trial_")  # Has unique suffix

    def test_job_metadata_creation(self, scheduler):
        """Test that jobs have proper metadata structure."""
        jobs = scheduler.schedule(runs=[], available_training_slots=1)

        job = jobs[0]
        assert isinstance(job.metadata, dict)
        assert job.created_at is not None

    def test_mixed_run_states(self, scheduler):
        """Test scheduling with runs in various states."""
        runs = [
            # Pending run (training not started)
            RunInfo(run_id="trial_0001", has_started_training=False),
            # Training in progress
            RunInfo(run_id="trial_0002", has_started_training=True, has_completed_training=False),
            # Training done, eval needed
            RunInfo(
                run_id="trial_0003", has_started_training=True, has_completed_training=True, has_started_eval=False
            ),
            # Eval in progress
            RunInfo(
                run_id="trial_0004",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=False,
            ),
            # Fully complete
            RunInfo(
                run_id="trial_0005",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
            ),
        ]

        jobs = scheduler.schedule(runs=runs, available_training_slots=1)

        # Should create: 1 eval job for trial_0003, no training jobs (at max_trials=3 with 5 runs)
        eval_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_EVAL]
        training_jobs = [j for j in jobs if j.type == JobTypes.LAUNCH_TRAINING]

        assert len(eval_jobs) == 1
        assert eval_jobs[0].run_id == "trial_0003"

        # No new training jobs should be created since we already have 5 runs >= max_trials(3)
        assert len(training_jobs) == 0
