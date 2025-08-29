"""Tests for sweep schedulers."""

from metta.sweep.scheduler.optimizing import OptimizingScheduler, OptimizingSchedulerConfig
from metta.sweep.scheduler.sequential import SequentialScheduler, SequentialSchedulerConfig
from metta.sweep.sweep_orchestrator import (
    JobTypes,
    Observation,
    RunInfo,
    SweepMetadata,
)


class TestOptimizingScheduler:
    """Test OptimizingScheduler with optimizer integration."""

    def test_optimizing_scheduler_initialization(self):
        """Test that OptimizingScheduler initializes correctly."""
        from metta.sweep.optimizer.protein import ProteinOptimizer
        from metta.sweep.protein_config import ParameterConfig, ProteinConfig

        # Use proper ParameterConfig like in standard.py
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
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
        config = OptimizingSchedulerConfig(
            max_trials=10,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = OptimizingScheduler(config, optimizer)

        assert scheduler.config.max_trials == 10
        assert scheduler.optimizer is not None
        assert len(scheduler._created_runs) == 0
        assert scheduler._is_complete is False

    def test_optimizing_scheduler_suggest_flow(self):
        """Test that OptimizingScheduler gets suggestions from optimizer."""
        from metta.sweep.optimizer.protein import ProteinOptimizer
        from metta.sweep.protein_config import ParameterConfig, ProteinConfig

        # Create protein config with proper ParameterConfig
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
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
        config = OptimizingSchedulerConfig(
            max_trials=3,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = OptimizingScheduler(config, optimizer)

        # Create sweep metadata
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Schedule first job
        jobs = scheduler.schedule(metadata, [])

        assert len(jobs) == 1
        assert jobs[0].type == JobTypes.LAUNCH_TRAINING
        assert "learning_rate" in jobs[0].config
        assert len(scheduler._created_runs) == 1

    def test_optimizing_scheduler_completion_detection(self):
        """Test that OptimizingScheduler detects when all trials are complete."""
        from metta.sweep.optimizer.protein import ProteinOptimizer
        from metta.sweep.protein_config import ParameterConfig, ProteinConfig

        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
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
        config = OptimizingSchedulerConfig(
            max_trials=2,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = OptimizingScheduler(config, optimizer)
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Schedule all trials
        jobs1 = scheduler.schedule(metadata, [])
        assert len(jobs1) == 1
        assert scheduler._is_complete is False

        # Simulate first job completed with evaluation
        run1 = RunInfo(
            run_id=jobs1[0].run_id,
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.5, cost=100, suggestion=jobs1[0].config),
        )

        # Add run to created_runs to track it
        scheduler._created_runs.add(run1.run_id)

        jobs2 = scheduler.schedule(metadata, [run1])
        assert len(jobs2) == 1
        assert scheduler._is_complete is False

        # Simulate second job completed with evaluation
        run2 = RunInfo(
            run_id=jobs2[0].run_id,
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.7, cost=100, suggestion=jobs2[0].config),
        )

        # Add run to created_runs to track it
        scheduler._created_runs.add(run2.run_id)

        # Should return no more jobs - max_trials reached
        jobs3 = scheduler.schedule(metadata, [run1, run2])
        assert len(jobs3) == 0
        assert scheduler._is_complete is True
        assert scheduler.is_complete is True


class TestSequentialScheduler:
    """Test SequentialScheduler implementation."""

    def test_sequential_scheduler_initialization(self):
        """Test that SequentialScheduler initializes correctly."""
        config = SequentialSchedulerConfig(
            max_trials=5,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = SequentialScheduler(config)

        assert scheduler.config.max_trials == 5
        assert scheduler._trial_count == 0
        assert scheduler._initialized is False

    def test_sequential_scheduler_one_at_time(self):
        """Test that SequentialScheduler only schedules one job at a time."""
        config = SequentialSchedulerConfig(
            max_trials=3,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = SequentialScheduler(config)
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Initialize with first job
        initial_jobs = scheduler.initialize("test_sweep")
        assert len(initial_jobs) == 1
        assert initial_jobs[0].type == JobTypes.LAUNCH_TRAINING

        # Simulate job is running - should not schedule more
        running_run = RunInfo(
            run_id=initial_jobs[0].run_id,
            has_started_training=True,
            has_completed_training=False,
        )

        jobs = scheduler.schedule(metadata, [running_run])
        assert len(jobs) == 0  # No new jobs while one is running

        # Simulate job completed - should schedule next
        completed_run = RunInfo(
            run_id=initial_jobs[0].run_id,
            has_started_training=True,
            has_completed_training=True,
            has_been_evaluated=True,
        )

        jobs = scheduler.schedule(metadata, [completed_run])
        assert len(jobs) == 1  # Next job scheduled
        assert scheduler._trial_count == 2

    def test_sequential_scheduler_max_trials(self):
        """Test that SequentialScheduler respects max_trials limit."""
        config = SequentialSchedulerConfig(
            max_trials=2,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = SequentialScheduler(config)
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Initialize
        initial_jobs = scheduler.initialize("test_sweep")
        assert scheduler._trial_count == 1

        # Complete first job and get second
        completed_run1 = RunInfo(
            run_id=initial_jobs[0].run_id,
            has_started_training=True,
            has_completed_training=True,
            has_been_evaluated=True,
        )

        jobs2 = scheduler.schedule(metadata, [completed_run1])
        assert len(jobs2) == 1
        assert scheduler._trial_count == 2

        # Complete second job - should not schedule more (max_trials=2)
        completed_run2 = RunInfo(
            run_id=jobs2[0].run_id,
            has_started_training=True,
            has_completed_training=True,
            has_been_evaluated=True,
        )

        jobs3 = scheduler.schedule(metadata, [completed_run1, completed_run2])
        assert len(jobs3) == 0  # Max trials reached

    def test_sequential_scheduler_eval_scheduling(self):
        """Test that SequentialScheduler schedules evaluations for completed training."""
        config = SequentialSchedulerConfig(
            max_trials=2,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = SequentialScheduler(config)
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Initialize
        scheduler.initialize("test_sweep")

        # Create a run that needs evaluation
        needs_eval_run = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
        )

        jobs = scheduler.schedule(metadata, [needs_eval_run])

        # Should schedule evaluation job
        assert len(jobs) == 1
        assert jobs[0].type == JobTypes.LAUNCH_EVAL
        assert jobs[0].run_id == "test_sweep_trial_0001_eval"
        assert "policy_uri" in jobs[0].overrides
