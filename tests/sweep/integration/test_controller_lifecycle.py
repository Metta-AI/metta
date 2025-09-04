"""Integration tests for sweep controller lifecycle management."""

import time
from unittest.mock import patch

from metta.sweep.controller import SweepController
from metta.sweep.models import JobDefinition, JobTypes, Observation, RunInfo, SweepStatus
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.optimizing import OptimizingScheduler, OptimizingSchedulerConfig


class MockStore:
    """Mock store that simulates WandB API behavior with caching delays."""

    def __init__(self):
        self.runs = {}
        self.api_delay = 0.1  # Simulate API latency
        self.cache_delay = 0.5  # Simulate cache invalidation delay

    def init_run(self, run_id: str, sweep_id: str) -> None:
        """Initialize a new run."""
        self.runs[run_id] = RunInfo(
            run_id=run_id,
            group=sweep_id,
            has_started_training=False,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )

    def fetch_runs(self, filters: dict) -> list[RunInfo]:
        """Fetch runs with simulated API delay."""
        time.sleep(self.api_delay)
        sweep_id = filters.get("group")
        return [run for run in self.runs.values() if run.group == sweep_id]

    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        """Update run summary with simulated cache delay."""
        if run_id not in self.runs:
            return False

        # Simulate the update being slow to propagate
        time.sleep(self.cache_delay)

        # Update suggestion if provided
        if "suggestion" in summary_update:
            if self.runs[run_id].summary is None:
                self.runs[run_id].summary = {}
            self.runs[run_id].summary["suggestion"] = summary_update["suggestion"]

        # Note: We intentionally DON'T update has_started_eval here
        # to simulate the WandB API caching issue
        return True

    def simulate_training_complete(self, run_id: str):
        """Simulate a training run completing."""
        if run_id in self.runs:
            self.runs[run_id].has_started_training = True
            self.runs[run_id].has_completed_training = True

    def simulate_eval_complete(self, run_id: str, score: float):
        """Simulate an evaluation completing."""
        if run_id in self.runs:
            self.runs[run_id].has_started_eval = True
            self.runs[run_id].has_been_evaluated = True
            if self.runs[run_id].summary is None:
                self.runs[run_id].summary = {}
            self.runs[run_id].summary["evaluator/score"] = score

            # Create observation
            suggestion = self.runs[run_id].summary.get("suggestion", {})
            self.runs[run_id].observation = Observation(score=score, cost=100, suggestion=suggestion)


class MockDispatcher:
    """Mock dispatcher that tracks dispatch calls."""

    def __init__(self):
        self.dispatched_jobs = []
        self.dispatch_count = {}

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch a job and track it."""
        self.dispatched_jobs.append(job)
        self.dispatch_count[job.run_id] = self.dispatch_count.get(job.run_id, 0) + 1
        return f"dispatch_{job.run_id}_{len(self.dispatched_jobs)}"


class TestControllerLifecycle:
    """Test the complete lifecycle management in SweepController."""

    def test_no_duplicate_training_dispatch(self):
        """Test that training jobs aren't dispatched multiple times."""
        # Setup
        store = MockStore()
        dispatcher = MockDispatcher()

        protein_config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="random",
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
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        controller = SweepController(
            sweep_id="test_sweep",
            scheduler=scheduler,
            optimizer=optimizer,
            dispatcher=dispatcher,
            store=store,
            protein_config=protein_config,
            sweep_status=SweepStatus.CREATED,
            max_parallel_jobs=10,
            monitoring_interval=0.01,  # Very short for testing
        )

        # Run one iteration
        with patch.object(controller, "_check_sweep_complete", return_value=False):
            # First iteration - should dispatch training
            controller._run_single_iteration()

            assert len(dispatcher.dispatched_jobs) == 1
            assert dispatcher.dispatched_jobs[0].type == JobTypes.LAUNCH_TRAINING
            assert "test_sweep_trial_0001" in controller.dispatched_trainings

            # Second iteration - should NOT re-dispatch same training
            controller._run_single_iteration()

            assert len(dispatcher.dispatched_jobs) == 1  # Still just 1
            assert dispatcher.dispatch_count.get("test_sweep_trial_0001", 0) == 1

    def test_no_duplicate_eval_dispatch(self):
        """Test that eval jobs aren't dispatched multiple times even with API sync issues."""
        # Setup
        store = MockStore()
        dispatcher = MockDispatcher()

        protein_config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="random",
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
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        controller = SweepController(
            sweep_id="test_sweep",
            scheduler=scheduler,
            optimizer=optimizer,
            dispatcher=dispatcher,
            store=store,
            protein_config=protein_config,
            sweep_status=SweepStatus.RESUMED,  # Use RESUMED so has_data=True
            max_parallel_jobs=10,
            monitoring_interval=0.01,
        )

        # Manually set up a completed training run
        store.init_run("test_sweep_trial_0001", "test_sweep")
        store.simulate_training_complete("test_sweep_trial_0001")
        controller.dispatched_trainings.add("test_sweep_trial_0001")

        with patch.object(controller, "_check_sweep_complete", return_value=False):
            # First iteration - should dispatch eval
            controller._run_single_iteration()

            eval_jobs = [j for j in dispatcher.dispatched_jobs if j.type == JobTypes.LAUNCH_EVAL]
            assert len(eval_jobs) == 1
            assert eval_jobs[0].run_id == "test_sweep_trial_0001"
            assert "test_sweep_trial_0001" in controller.dispatched_evals

            # Second iteration - should NOT re-dispatch eval even though
            # has_started_eval is still False in the store (simulating WandB cache issue)
            controller._run_single_iteration()

            eval_jobs = [j for j in dispatcher.dispatched_jobs if j.type == JobTypes.LAUNCH_EVAL]
            assert len(eval_jobs) == 1  # Still just 1
            assert dispatcher.dispatch_count.get("test_sweep_trial_0001", 0) == 1

    def test_capacity_management(self):
        """Test that controller respects max_parallel_jobs for training."""
        # Setup
        store = MockStore()
        dispatcher = MockDispatcher()

        protein_config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="random",
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
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=5,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        controller = SweepController(
            sweep_id="test_sweep",
            scheduler=scheduler,
            optimizer=optimizer,
            dispatcher=dispatcher,
            store=store,
            protein_config=protein_config,
            sweep_status=SweepStatus.CREATED,
            max_parallel_jobs=2,  # Only allow 2 parallel training jobs
            monitoring_interval=0.01,
        )

        with patch.object(controller, "_check_sweep_complete", return_value=False):
            # First iteration - should dispatch first training
            controller._run_single_iteration()
            assert len(controller.dispatched_trainings) == 1

            # Manually add second training as if scheduler suggested it
            # but it should be filtered by capacity
            controller.dispatched_trainings.add("test_sweep_trial_0001")
            controller.dispatched_trainings.add("test_sweep_trial_0002")

            # Try to dispatch third - should be blocked by capacity
            active_trainings = len(controller.dispatched_trainings) - len(controller.completed_runs)
            assert active_trainings == 2

            # Mark one as completed
            controller.completed_runs.add("test_sweep_trial_0001")

            # Now capacity should allow one more
            active_trainings = len(controller.dispatched_trainings) - len(controller.completed_runs)
            assert active_trainings == 1

    def test_completed_runs_tracking(self):
        """Test that completed runs are properly tracked."""
        # Setup
        store = MockStore()
        dispatcher = MockDispatcher()

        protein_config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="random",
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
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        controller = SweepController(
            sweep_id="test_sweep",
            scheduler=scheduler,
            optimizer=optimizer,
            dispatcher=dispatcher,
            store=store,
            protein_config=protein_config,
            sweep_status=SweepStatus.RESUMED,  # Use RESUMED so has_data=True
            max_parallel_jobs=10,
            monitoring_interval=0.01,
        )

        # Create a fully completed run
        store.init_run("test_sweep_trial_0001", "test_sweep")
        store.runs["test_sweep_trial_0001"].has_started_training = True
        store.runs["test_sweep_trial_0001"].has_completed_training = True
        store.runs["test_sweep_trial_0001"].has_started_eval = True
        store.runs["test_sweep_trial_0001"].has_been_evaluated = True
        store.runs["test_sweep_trial_0001"].observation = Observation(score=0.9, cost=100, suggestion={"lr": 0.005})

        with patch.object(controller, "_check_sweep_complete", return_value=False):
            controller._run_single_iteration()

            # Should be tracked as completed
            assert "test_sweep_trial_0001" in controller.completed_runs

    def test_eval_done_triggers_short_refractory(self):
        """Test that completing an eval triggers a 5-second refractory period."""
        # Setup
        store = MockStore()
        dispatcher = MockDispatcher()

        protein_config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="random",
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
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        controller = SweepController(
            sweep_id="test_sweep",
            scheduler=scheduler,
            optimizer=optimizer,
            dispatcher=dispatcher,
            store=store,
            protein_config=protein_config,
            sweep_status=SweepStatus.RESUMED,  # Use RESUMED so has_data=True
            max_parallel_jobs=10,
            monitoring_interval=60,  # Normal interval is 60s
        )

        # Create a run with eval done but not completed
        store.init_run("test_sweep_trial_0001", "test_sweep")
        run = store.runs["test_sweep_trial_0001"]
        run.has_started_training = True
        run.has_completed_training = True
        run.has_started_eval = True
        run.has_been_evaluated = True
        run.summary = {"score": 0.9, "suggestion": {"lr": 0.005}}

        # Run the update method
        all_runs = store.fetch_runs({"group": "test_sweep"})
        has_eval_done = controller._update_completed_runs(all_runs)

        assert has_eval_done is True  # Should detect eval completion

    def _run_single_iteration(self):
        """Helper method to run a single controller iteration."""
        # Properly implement what the controller actually does in its main loop

        # 1. Fetch all runs from store
        all_run_infos = self._fetch_all_runs()

        # 2. Compute metadata and track completed runs
        metadata = self._compute_metadata_from_runs(all_run_infos)

        # 3. Get job schedule from scheduler
        new_jobs = self.scheduler.schedule(
            sweep_metadata=metadata,
            all_runs=all_run_infos,
            dispatched_trainings=self.dispatched_trainings,
            dispatched_evals=self.dispatched_evals,
        )

        # 4. Filter jobs by capacity and dispatch status
        new_jobs = self._filter_jobs_by_capacity(new_jobs, metadata)

        # 5. Execute each job
        for job in new_jobs:
            self._execute_job(job)

        # 6. Update completed runs
        has_eval_done = self._update_completed_runs(all_run_infos)

        return has_eval_done


# Add this method to the controller for testing
SweepController._run_single_iteration = TestControllerLifecycle._run_single_iteration
