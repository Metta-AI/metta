"""Tests for sweep schedulers."""

from metta.sweep import JobTypes, Observation, RunInfo, SweepMetadata
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.optimizing import OptimizingScheduler, OptimizingSchedulerConfig


class TestOptimizingScheduler:
    """Test OptimizingScheduler with optimizer integration."""

    def test_optimizing_scheduler_initialization(self):
        """Test that OptimizingScheduler initializes correctly."""
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
