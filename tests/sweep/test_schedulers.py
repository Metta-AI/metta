"""Tests for sweep schedulers."""

from metta.sweep.models import JobTypes, Observation, RunInfo, SweepMetadata
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.optimizing import OptimizingScheduler, OptimizingSchedulerConfig


class TestOptimizingScheduler:
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
        # Scheduler no longer tracks created runs internally
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

        # Schedule first job - no dispatched jobs yet
        jobs = scheduler.schedule(metadata, [], set(), set())

        assert len(jobs) == 1
        assert jobs[0].type == JobTypes.LAUNCH_TRAINING
        assert jobs[0].run_id == "test_sweep_trial_0001"
        assert "learning_rate" in jobs[0].config  # Should have optimizer suggestion

    def test_optimizing_scheduler_schedules_eval(self):
        """Test that scheduler schedules evaluation for completed training."""
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
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Simulate a training run that needs evaluation
        run_needing_eval = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )

        # Schedule should return an eval job
        jobs = scheduler.schedule(
            metadata,
            [run_needing_eval],
            {"test_sweep_trial_0001"},  # Training was dispatched
            set(),  # No eval dispatched yet
        )

        assert len(jobs) == 1
        assert jobs[0].type == JobTypes.LAUNCH_EVAL
        assert jobs[0].run_id == "test_sweep_trial_0001"

    def test_optimizing_scheduler_skips_already_dispatched_eval(self):
        """Test that scheduler doesn't reschedule already dispatched evals."""
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
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Simulate a training run that needs evaluation
        run_needing_eval = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )

        # Schedule with eval already dispatched
        jobs = scheduler.schedule(
            metadata,
            [run_needing_eval],
            {"test_sweep_trial_0001"},  # Training was dispatched
            {"test_sweep_trial_0001"},  # Eval also already dispatched
        )

        # Should return empty list since eval already dispatched
        assert len(jobs) == 0

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

        # Schedule first trial
        jobs1 = scheduler.schedule(metadata, [], set(), set())
        assert len(jobs1) == 1
        assert scheduler._is_complete is False

        # Simulate first job completed with evaluation
        run1 = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.5, cost=100, suggestion={"learning_rate": 0.005}),
        )

        # Schedule second trial
        jobs2 = scheduler.schedule(
            metadata,
            [run1],
            {"test_sweep_trial_0001"},  # First trial dispatched
            {"test_sweep_trial_0001"},  # First trial eval dispatched
        )
        assert len(jobs2) == 1
        assert jobs2[0].run_id == "test_sweep_trial_0002"
        assert scheduler._is_complete is False

        # Simulate second job completed with evaluation
        run2 = RunInfo(
            run_id="test_sweep_trial_0002",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.7, cost=100, suggestion={"learning_rate": 0.003}),
        )

        # Should return no more jobs - max_trials reached
        jobs3 = scheduler.schedule(
            metadata,
            [run1, run2],
            {"test_sweep_trial_0001", "test_sweep_trial_0002"},  # Both trials dispatched
            {"test_sweep_trial_0001", "test_sweep_trial_0002"},  # Both evals dispatched
        )
        assert len(jobs3) == 0
        assert scheduler._is_complete is True
        assert scheduler.is_complete is True

    def test_optimizing_scheduler_avoids_duplicate_run_ids(self):
        """Test that scheduler doesn't create duplicate run IDs."""
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
            max_trials=5,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )

        scheduler = OptimizingScheduler(config, optimizer)
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Simulate that trial_0001 is already dispatched
        jobs = scheduler.schedule(
            metadata,
            [],
            {"test_sweep_trial_0001"},  # Already dispatched
            set(),
        )

        # Should get trial_0002 since trial_0001 is already dispatched
        assert len(jobs) == 1
        assert jobs[0].run_id == "test_sweep_trial_0002"

    def test_optimizing_scheduler_respects_max_trials(self):
        """Test that scheduler stops scheduling after max_trials."""
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

        # Simulate that we've already dispatched max_trials
        dispatched_trainings = {"test_sweep_trial_0001", "test_sweep_trial_0002"}

        # Create completed runs
        run1 = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.5, cost=100, suggestion={"learning_rate": 0.005}),
        )

        run2 = RunInfo(
            run_id="test_sweep_trial_0002",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
            observation=Observation(score=0.7, cost=100, suggestion={"learning_rate": 0.003}),
        )

        # Try to schedule more - should return empty and mark as complete
        jobs = scheduler.schedule(
            metadata, [run1, run2], dispatched_trainings, {"test_sweep_trial_0001", "test_sweep_trial_0002"}
        )

        assert len(jobs) == 0
        assert scheduler._is_complete is True

    def test_optimizing_scheduler_waits_for_incomplete_jobs(self):
        """Test that scheduler waits for incomplete jobs before scheduling new ones."""
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
        metadata = SweepMetadata(sweep_id="test_sweep")

        # Simulate a run that's still in training
        run_in_training = RunInfo(
            run_id="test_sweep_trial_0001",
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )

        # Schedule should wait and not create new jobs
        jobs = scheduler.schedule(metadata, [run_in_training], {"test_sweep_trial_0001"}, set())

        assert len(jobs) == 0  # Should wait for training to complete
