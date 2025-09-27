"""Tests for BatchedSyncedOptimizingScheduler basic functionality."""

from datetime import datetime, timezone

from softmax.training.adaptive.models import JobTypes, RunInfo
from softmax.training.sweep.protein_config import ParameterConfig, ProteinConfig
from softmax.training.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
)


class TestBatchedSyncedOptimizingScheduler:
    """Test suite for BatchedSyncedOptimizingScheduler basic functionality."""

    def test_initialization(self):
        """Test scheduler initialization."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "learning_rate": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=20,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_exp",
            protein_config=protein_config,
        )

        scheduler = BatchedSyncedOptimizingScheduler(config)

        assert scheduler.config.max_trials == 20
        assert scheduler.config.batch_size == 4  # default batch size
        assert scheduler.optimizer is not None

    def test_batch_generation_when_all_complete(self):
        """Test that scheduler generates batch only when all runs are complete."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            batch_size=2,
            experiment_id="test_sweep",
            protein_config=protein_config,
        )

        scheduler = BatchedSyncedOptimizingScheduler(config)

        # Case 1: No existing runs - should generate batch
        jobs = scheduler.schedule([], available_training_slots=5)

        assert len(jobs) == 2  # Should generate batch of 2
        assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)
        assert jobs[0].run_id.startswith("test_sweep_trial_0001_")
        assert jobs[1].run_id.startswith("test_sweep_trial_0002_")

    def test_wait_for_incomplete_runs(self):
        """Test that scheduler waits for all runs to complete before next batch."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            batch_size=3,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_sweep",
            protein_config=protein_config,
        )
        scheduler = BatchedSyncedOptimizingScheduler(config)

        # Simulate that we have runs in training
        scheduler.state.runs_in_training = {"test_sweep_trial_0001", "test_sweep_trial_0002"}

        # Create runs with mixed statuses
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=False,  # Still training
                has_started_eval=False,
                has_been_evaluated=False,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=False,  # Still evaluating
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            ),
        ]

        # Should not generate new batch while runs are incomplete
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0  # Should wait for all to complete

    def test_schedule_evaluations(self):
        """Test that scheduler schedules evaluations for completed training."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            batch_size=3,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_sweep",
            protein_config=protein_config,
        )
        scheduler = BatchedSyncedOptimizingScheduler(config)

        # Mark runs as in training (so eval scheduler knows about them)
        scheduler.state.runs_in_training = {"test_sweep_trial_0001", "test_sweep_trial_0002"}

        # Create runs that need evaluation
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,  # Needs evaluation
                has_been_evaluated=False,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,  # Needs evaluation
                has_been_evaluated=False,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            ),
        ]

        # Should schedule evaluations
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 2  # Should schedule both evaluations
        assert all(job.type == JobTypes.LAUNCH_EVAL for job in jobs)
        assert jobs[0].run_id == "test_sweep_trial_0001"
        assert jobs[1].run_id == "test_sweep_trial_0002"

    def test_no_duplicate_eval_scheduling(self):
        """Test that scheduler doesn't reschedule already dispatched evaluations."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            batch_size=3,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_sweep",
            protein_config=protein_config,
        )
        scheduler = BatchedSyncedOptimizingScheduler(config)

        # Mark run as already in eval
        scheduler.state.runs_in_eval = {"test_sweep_trial_0001"}

        # Create run that appears to need evaluation
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,  # Looks like it needs eval
                has_been_evaluated=False,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            ),
        ]

        # Should not reschedule since already in eval
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 0  # Should not reschedule

    def test_max_trials_handling(self):
        """Test that scheduler respects max_trials limit."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=5,
            batch_size=3,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_sweep",
            protein_config=protein_config,
        )
        scheduler = BatchedSyncedOptimizingScheduler(config)

        # All runs completed
        runs = [
            RunInfo(
                run_id=f"test_sweep_trial_{i:04d}",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
                summary={
                    "sweep/score": 0.5 + i * 0.1,
                    "sweep/cost": 100,
                    "sweep/suggestion": {"lr": 0.005},
                },
            )
            for i in range(1, 4)
        ]

        # Mark these as completed in state
        scheduler.state.runs_completed = {f"test_sweep_trial_{i:04d}" for i in range(1, 4)}

        # Should only generate 2 more to reach max_trials=5
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 2  # Only 2 more to reach limit
        assert jobs[0].run_id.startswith("test_sweep_trial_0004_")
        assert jobs[1].run_id.startswith("test_sweep_trial_0005_")

    def test_batch_with_observations(self):
        """Test that scheduler uses observations when generating new batch."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "lr": ParameterConfig(
                    min=0.001,
                    max=0.01,
                    distribution="log_normal",
                    mean=0.003,
                    scale="auto",
                )
            },
        )

        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            batch_size=2,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            experiment_id="test_sweep",
            protein_config=protein_config,
        )
        scheduler = BatchedSyncedOptimizingScheduler(config)

        # Previous completed runs with observations in summary
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
                summary={
                    "sweep/score": 0.5,
                    "sweep/cost": 100,
                    "sweep/suggestion": {"lr": 0.005},
                },
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                has_failed=False,
                created_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
                summary={
                    "sweep/score": 0.8,
                    "sweep/cost": 100,
                    "sweep/suggestion": {"lr": 0.003},
                },
            ),
        ]

        # Mark as completed in state
        scheduler.state.runs_completed = {"test_sweep_trial_0001", "test_sweep_trial_0002"}

        # Should generate new batch using observations
        jobs = scheduler.schedule(runs, available_training_slots=5)

        assert len(jobs) == 2  # Batch of 2
        assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)
        # The jobs should have suggestions in metadata
        assert all("sweep/suggestion" in job.metadata for job in jobs)
        assert all("lr" in job.metadata["sweep/suggestion"] for job in jobs)
