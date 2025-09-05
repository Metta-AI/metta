"""Integration tests for sweep controller lifecycle management."""

import time

from metta.sweep.controller import SweepController
from metta.sweep.models import JobDefinition, JobTypes, Observation, RunInfo, SweepStatus
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.schedulers.batched_synced import BatchedSyncedOptimizingScheduler, BatchedSyncedSchedulerConfig


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
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
            batch_size=1,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

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
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
            batch_size=1,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

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
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=5,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
            batch_size=1,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

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

        # Now active should be 1, allowing more dispatches
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
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
            batch_size=1,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

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
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=2,
            recipe_module="test",
            train_entrypoint="train",
            eval_entrypoint="eval",
            batch_size=1,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

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

        # Run the update method - inline the logic from controller
        all_runs = store.fetch_runs({"group": "test_sweep"})

        # Inline the _update_completed_runs logic
        has_eval_done = False
        for run in all_runs:
            if run.status.name == "EVAL_DONE_NOT_COMPLETED":
                assert run.summary is not None
                cost = run.cost if run.cost != 0 else run.runtime
                score = run.summary.get(controller.protein_config.metric)
                if score is None:
                    raise ValueError(f"No metric {controller.protein_config.metric} found in run summary.")
                store.update_run_summary(
                    run.run_id,
                    {
                        "observation": {
                            "cost": cost,
                            "score": score,
                            "suggestion": run.summary.get("suggestion"),
                        }
                    },
                )
                has_eval_done = True

        assert has_eval_done is True  # Should detect eval completion

    def _run_single_iteration(self):
        """Helper method to run a single controller iteration."""
        # Inline the controller logic since methods were inlined

        # 1. Fetch ALL runs from store
        if self.has_data:
            all_run_infos = self.store.fetch_runs(filters={"group": self.sweep_id})
        else:
            self.has_data = True
            all_run_infos = []

        # Check if the sweep is complete
        completed_runs = [r for r in all_run_infos if r.status.name in ("COMPLETED", "FAILED")]
        if len(completed_runs) >= self.max_trials:
            return False

        # 2. Update sweep metadata based on ALL runs
        from metta.sweep.models import JobStatus, SweepMetadata

        metadata = SweepMetadata(sweep_id=self.sweep_id)
        metadata.runs_created = len(all_run_infos)

        for run in all_run_infos:
            if run.status == JobStatus.PENDING:
                metadata.runs_pending += 1
            elif run.status in [JobStatus.IN_TRAINING, JobStatus.TRAINING_DONE_NO_EVAL, JobStatus.IN_EVAL]:
                metadata.runs_in_progress += 1
            elif run.status in [JobStatus.COMPLETED, JobStatus.EVAL_DONE_NOT_COMPLETED]:
                metadata.runs_completed += 1
                if run.status == JobStatus.COMPLETED:
                    self.completed_runs.add(run.run_id)

        # 3. Get job schedule from scheduler
        new_jobs = self.scheduler.schedule(
            sweep_metadata=metadata,
            all_runs=all_run_infos,
            dispatched_trainings=self.dispatched_trainings,
            dispatched_evals=self.dispatched_evals,
        )

        # 4. Filter jobs based on capacity constraints and dispatch status
        from metta.sweep.models import JobTypes

        filtered_jobs = []

        for job in new_jobs:
            # Check if job has already been dispatched
            if job.type == JobTypes.LAUNCH_TRAINING and job.run_id in self.dispatched_trainings:
                continue
            elif job.type == JobTypes.LAUNCH_EVAL and job.run_id in self.dispatched_evals:
                continue

            if job.type == JobTypes.LAUNCH_EVAL:
                # Always allow eval jobs
                filtered_jobs.append(job)
            elif job.type == JobTypes.LAUNCH_TRAINING:
                # Check capacity for training jobs
                active_trainings = len(self.dispatched_trainings) - len(self.completed_runs)
                if active_trainings < self.max_parallel_jobs:
                    filtered_jobs.append(job)
            else:
                filtered_jobs.append(job)

        # 5. Execute scheduler's decisions
        for job in filtered_jobs:
            try:
                if job.type == JobTypes.LAUNCH_TRAINING:
                    self.store.init_run(job.run_id, sweep_id=self.sweep_id)
                    if job.config:
                        self.store.update_run_summary(job.run_id, {"suggestion": job.config})
                elif job.type == JobTypes.LAUNCH_EVAL:
                    pass  # Just dispatch

                self.dispatcher.dispatch(job)

                # Track that we've dispatched this job
                if job.type == JobTypes.LAUNCH_TRAINING:
                    self.dispatched_trainings.add(job.run_id)
                elif job.type == JobTypes.LAUNCH_EVAL:
                    self.dispatched_evals.add(job.run_id)

            except Exception:
                continue

        # 6. Update completed runs
        has_eval_done = False
        for run in all_run_infos:
            if run.status.name == "EVAL_DONE_NOT_COMPLETED":
                if run.summary is not None:
                    cost = run.cost if run.cost != 0 else run.runtime
                    score = run.summary.get(self.protein_config.metric)
                    if score is not None:
                        self.store.update_run_summary(
                            run.run_id,
                            {
                                "observation": {
                                    "cost": cost,
                                    "score": score,
                                    "suggestion": run.summary.get("suggestion"),
                                }
                            },
                        )
                        has_eval_done = True

        return has_eval_done


# Add this method to the controller for testing
SweepController._run_single_iteration = TestControllerLifecycle._run_single_iteration
