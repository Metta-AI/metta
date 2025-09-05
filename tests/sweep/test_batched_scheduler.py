"""Tests for BatchedSyncedOptimizingScheduler."""

from metta.sweep.models import JobTypes, Observation, RunInfo, SweepMetadata
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
)


class TestBatchedSyncedOptimizingScheduler:
    """Test suite for BatchedSyncedOptimizingScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(
            max_trials=20,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        assert scheduler.config.max_trials == 20
        assert scheduler.config.batch_size == 4  # default batch size

    def test_batch_generation_when_all_complete(self):
        """Test that scheduler generates batch only when all runs are complete."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(
            max_trials=10,
            recipe_module="test.module",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
            batch_size=2,
        )

        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Case 1: No existing runs - should generate batch
        jobs = scheduler.schedule(metadata, [], set(), set())

        assert len(jobs) == 2  # Should generate batch of 2
        assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)
        assert jobs[0].run_id == "test_sweep_trial_0001"
        assert jobs[1].run_id == "test_sweep_trial_0002"

    def test_wait_for_incomplete_runs(self):
        """Test that scheduler waits for all runs to complete before next batch."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(max_trials=10)
        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Create runs with mixed statuses
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                observation=Observation(score=0.5, cost=100, suggestion={"lr": 0.005}),
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=False,  # Still training
            ),
            RunInfo(
                run_id="test_sweep_trial_0003",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=False,  # Still evaluating
            ),
        ]

        # Should not generate new batch while runs are incomplete
        jobs = scheduler.schedule(
            metadata,
            runs,
            {"test_sweep_trial_0001", "test_sweep_trial_0002", "test_sweep_trial_0003"},
            {"test_sweep_trial_0001", "test_sweep_trial_0003"},
        )

        assert len(jobs) == 0  # Should wait for all to complete

    def test_schedule_evaluations(self):
        """Test that scheduler schedules evaluations for completed training."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(max_trials=10)
        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Create runs that need evaluation
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,  # Needs evaluation
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,  # Needs evaluation
            ),
        ]

        # Should schedule evaluations
        jobs = scheduler.schedule(
            metadata,
            runs,
            {"test_sweep_trial_0001", "test_sweep_trial_0002"},
            set(),  # No evals dispatched yet
        )

        assert len(jobs) == 2  # Should schedule both evaluations
        assert all(job.type == JobTypes.LAUNCH_EVAL for job in jobs)
        assert jobs[0].run_id == "test_sweep_trial_0001"
        assert jobs[1].run_id == "test_sweep_trial_0002"

    def test_no_duplicate_eval_scheduling(self):
        """Test that scheduler doesn't reschedule already dispatched evaluations."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(max_trials=10)
        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Create run that needs evaluation
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=False,
            ),
        ]

        # Evaluation already dispatched
        jobs = scheduler.schedule(
            metadata,
            runs,
            {"test_sweep_trial_0001"},
            {"test_sweep_trial_0001"},  # Already dispatched
        )

        assert len(jobs) == 0  # Should not reschedule

    def test_max_trials_handling(self):
        """Test that scheduler respects max_trials limit."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(max_trials=5, batch_size=3)
        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Already have 3 dispatched trainings
        dispatched_trainings = {
            "test_sweep_trial_0001",
            "test_sweep_trial_0002",
            "test_sweep_trial_0003",
        }

        # All runs completed
        runs = [
            RunInfo(
                run_id=f"test_sweep_trial_{i:04d}",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                observation=Observation(score=0.5, cost=100, suggestion={"lr": 0.005}),
            )
            for i in range(1, 4)
        ]

        # Should only generate 2 more to reach max_trials=5
        jobs = scheduler.schedule(
            metadata,
            runs,
            dispatched_trainings,
            dispatched_trainings,  # All evaluated
        )

        assert len(jobs) == 2  # Only 2 more to reach limit
        assert jobs[0].run_id == "test_sweep_trial_0004"
        assert jobs[1].run_id == "test_sweep_trial_0005"

    def test_batch_with_observations(self):
        """Test that scheduler uses observations when generating new batch."""
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",  # Use Bayesian optimization
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

        optimizer = ProteinOptimizer(protein_config)
        config = BatchedSyncedSchedulerConfig(max_trials=10, batch_size=2)
        scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)

        metadata = SweepMetadata(sweep_id="test_sweep")

        # Previous completed runs with observations
        runs = [
            RunInfo(
                run_id="test_sweep_trial_0001",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                observation=Observation(score=0.5, cost=100, suggestion={"lr": 0.005}),
            ),
            RunInfo(
                run_id="test_sweep_trial_0002",
                has_started_training=True,
                has_completed_training=True,
                has_started_eval=True,
                has_been_evaluated=True,
                observation=Observation(score=0.8, cost=100, suggestion={"lr": 0.003}),
            ),
        ]

        # Should generate new batch using observations
        jobs = scheduler.schedule(
            metadata,
            runs,
            {"test_sweep_trial_0001", "test_sweep_trial_0002"},
            {"test_sweep_trial_0001", "test_sweep_trial_0002"},
        )

        assert len(jobs) == 2  # Batch of 2
        assert all(job.type == JobTypes.LAUNCH_TRAINING for job in jobs)
        # The optimizer should have received the observations
        assert all("lr" in job.config for job in jobs)
