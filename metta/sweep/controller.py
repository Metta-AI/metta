"""Sweep controller and orchestration logic."""

import logging
import time
from dataclasses import dataclass

from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.models import JobStatus, JobTypes, RunInfo, SweepMetadata, SweepStatus
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protocols import Dispatcher, Optimizer, Scheduler, Store
from metta.sweep.utils import make_monitor_table

logger = logging.getLogger(__name__)


class SweepController:
    """Stateless orchestrator for sweep execution."""

    def __init__(
        self,
        sweep_id: str,
        scheduler: Scheduler,
        optimizer: Optimizer,
        dispatcher: Dispatcher,
        store: Store,
        protein_config: ProteinConfig,
        sweep_status: SweepStatus = SweepStatus.RESUMED,
        max_parallel_jobs: int = 10,
        monitoring_interval: int = 5,
    ):
        # Configuration only - no state
        self.sweep_id = sweep_id
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dispatcher = dispatcher
        self.store = store
        self.protein_config = protein_config
        self.monitoring_interval = monitoring_interval
        self.max_parallel_jobs = max_parallel_jobs
        self.has_data = sweep_status == SweepStatus.RESUMED
        self.dispatched_evals = set()  # Track evaluations we've dispatched
        self.dispatched_trainings = set()  # Track training jobs we've dispatched
        self.completed_runs = set()  # Track runs that have completed (training + eval)

    def _compute_metadata_from_runs(self, all_runs: list[RunInfo]) -> SweepMetadata:
        """Compute sweep metadata from runs."""
        metadata = SweepMetadata(sweep_id=self.sweep_id)
        metadata.runs_created = len(all_runs)  # Total number of runs

        for run in all_runs:
            if run.status == JobStatus.PENDING:
                metadata.runs_pending += 1
            elif run.status in [JobStatus.IN_TRAINING, JobStatus.TRAINING_DONE_NO_EVAL, JobStatus.IN_EVAL]:
                metadata.runs_in_progress += 1
            elif run.status in [JobStatus.COMPLETED, JobStatus.EVAL_DONE_NOT_COMPLETED]:
                metadata.runs_completed += 1
                # Track completed runs in-memory
                if run.status == JobStatus.COMPLETED:
                    self.completed_runs.add(run.run_id)

        return metadata

    def _fetch_all_runs(self) -> list[RunInfo]:
        """Fetch all runs from store."""
        if self.has_data:
            return self.store.fetch_runs(filters={"group": self.sweep_id})
        else:
            self.has_data = True  # We do this for the very first run
            # because WandB is weird about fetching empty sets.
            return []

    def _display_monitoring_table(self, all_run_infos: list[RunInfo]) -> None:
        """Display monitoring table if there are runs."""
        if all_run_infos:
            table_lines = make_monitor_table(
                runs=all_run_infos,
                title="Run Status Table",
                logger_prefix="[SweepController]",
                include_score=True,
                truncate_run_id=True,
            )
            for line in table_lines:
                logger.info(line)

    def _filter_jobs_by_capacity(self, new_jobs: list, metadata: SweepMetadata) -> list:
        # Filter jobs based on capacity constraints and dispatch status
        filtered_jobs = []

        for job in new_jobs:
            # Check if job has already been dispatched
            if job.type == JobTypes.LAUNCH_TRAINING and job.run_id in self.dispatched_trainings:
                logger.debug(f"[SweepController] Training job {job.run_id} already dispatched, skipping")
                continue
            elif job.type == JobTypes.LAUNCH_EVAL and job.run_id in self.dispatched_evals:
                logger.debug(f"[SweepController] Eval job {job.run_id} already dispatched, skipping")
                continue

            if job.type == JobTypes.LAUNCH_EVAL:
                # Always allow eval jobs (they don't consume cluster resources)
                filtered_jobs.append(job)
            elif job.type == JobTypes.LAUNCH_TRAINING:
                # Check if we have capacity for training jobs
                # Capacity = dispatched trainings - completed runs
                active_trainings = len(self.dispatched_trainings) - len(self.completed_runs)
                if active_trainings < self.max_parallel_jobs:
                    filtered_jobs.append(job)
                else:
                    logger.debug(
                        f"[SweepController] At max parallel jobs limit ({self.max_parallel_jobs}), "
                        f"skipping training job {job.run_id}"
                    )
            else:
                # Other job types (if any) go through
                filtered_jobs.append(job)

        return filtered_jobs

    def _execute_job(self, job) -> None:
        """Execute a single job."""
        try:
            if job.type == JobTypes.LAUNCH_TRAINING:
                self.store.init_run(job.run_id, sweep_id=self.sweep_id)
                # Store the suggestion (hyperparameters) in the run summary
                if job.config:  # job.config contains the optimizer suggestion
                    self.store.update_run_summary(job.run_id, {"suggestion": job.config})
                logger.info(f"[SweepController] Created run {job.run_id}")
            elif job.type == JobTypes.LAUNCH_EVAL:
                # Just dispatch the eval job - let the eval process itself update the status
                # This avoids WandB API caching issues with pre-dispatch status updates
                logger.info(f"[SweepController] Launching eval for job {job.run_id}")

            # Only dispatch if store operations succeeded
            dispatch_id = self.dispatcher.dispatch(job)
            logger.info(f"[SweepController] Dispatched {job.run_id} with dispatch_id {dispatch_id}")

            # Track that we've dispatched this job
            if job.type == JobTypes.LAUNCH_TRAINING:
                self.dispatched_trainings.add(job.run_id)
            elif job.type == JobTypes.LAUNCH_EVAL:
                self.dispatched_evals.add(job.run_id)

        except Exception as e:
            logger.error(f"[SweepController] Failed to initialize/dispatch job {job.run_id}: {e}")
            logger.error(f"[SweepController] Skipping dispatch for {job.run_id} to prevent resource overload")
            # Continue with next job rather than crashing the whole sweep

    def _update_completed_runs(self, all_run_infos: list[RunInfo]) -> bool:
        """Update transient states and mark completions.

        Returns:
            True if any eval was completed this cycle, False otherwise.
        """
        has_eval_done = False
        # TODO: Refactor: Sweep config and Optimizer config
        for run in all_run_infos:
            if run.status == JobStatus.EVAL_DONE_NOT_COMPLETED:
                assert run.summary is not None
                cost = run.cost if run.cost != 0 else run.runtime
                score = run.summary.get(self.protein_config.metric)
                if score is None:
                    raise ValueError(f"No metric {self.protein_config.metric} found in run summary.")
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
                # Mark that we had an eval complete for shorter refractory period
                has_eval_done = True

        return has_eval_done

    def _check_sweep_complete(self) -> bool:
        """Check if the sweep is complete."""
        # TODO: We should modify scheduler to always have that attribute.
        if hasattr(self.scheduler, "is_complete") and self.scheduler.is_complete:  # type: ignore
            logger.info("[SweepController]  Sweep completed successfully.")
            return True
        return False

    def run(self) -> None:
        """Main control loop for sweep execution."""
        while True:
            next_sleep = self.monitoring_interval  # Default monitoring interval
            try:
                # 1. Fetch ALL runs from store
                all_run_infos = self._fetch_all_runs()

                # 2. Update sweep metadata based on ALL runs
                metadata = self._compute_metadata_from_runs(all_run_infos)

                # Display monitoring table every interval
                self._display_monitoring_table(all_run_infos)

                # 3. Hand everything to scheduler - it decides what to do
                # Always get jobs from scheduler (including eval jobs)
                new_jobs = self.scheduler.schedule(
                    sweep_metadata=metadata,
                    all_runs=all_run_infos,
                    dispatched_trainings=self.dispatched_trainings,
                    dispatched_evals=self.dispatched_evals,
                )

                # Check if the sweep is complete
                if self._check_sweep_complete():
                    break

                # Filter jobs based on capacity constraints
                new_jobs = self._filter_jobs_by_capacity(new_jobs, metadata)

                # 4. Execute scheduler's decisions
                for job in new_jobs:
                    self._execute_job(job)

                # 5. Finally, update transient states and mark completions
                has_eval_done = self._update_completed_runs(all_run_infos)

                # 5. Sleep - use shorter interval if eval just completed
                if has_eval_done:
                    logger.debug("[SweepController] Using 5s refractory period after eval completion")
                    next_sleep = 5
                time.sleep(next_sleep)

            except KeyboardInterrupt:
                logger.info("[SweepController] Received interrupt signal, stopping sweep")
                break
            except Exception as e:
                logger.error(f"[SweepController] Error in control loop: {e}")
                time.sleep(self.monitoring_interval)


@dataclass
class SweepControllerConfig:
    sweep_name: str
    sweep_server_uri: str
    wandb: WandbConfig
    protein_config: ProteinConfig
    max_parallel_jobs: int = 10
    monitoring_interval: int = 60
