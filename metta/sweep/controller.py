"""Sweep controller and orchestration logic."""

import logging
import time
from dataclasses import dataclass

from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.models import JobStatus, JobTypes, RunInfo, SweepMetadata, SweepStatus
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protocols import Dispatcher, Optimizer, Scheduler, Store

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

        return metadata

    def run(self) -> None:
        """Main control loop for sweep execution."""
        while True:
            try:
                # 1. Fetch ALL runs from store
                if self.has_data:
                    all_run_infos = self.store.fetch_runs(filters={"group": self.sweep_id})  # Returns list[RunInfo]
                else:
                    all_run_infos = []
                    self.has_data = True  # We do this for the very first run
                    # beacause WandB is weird about fetching empty sets.

                # 2. Update sweep metadata based on ALL runs
                metadata = self._compute_metadata_from_runs(all_run_infos)

                # 3. Hand everything to scheduler - it decides what to do
                # Always get jobs from scheduler (including eval jobs)
                new_jobs = self.scheduler.schedule(sweep_metadata=metadata, all_runs=all_run_infos)

                # Check if the sweep is complete
                # TODO: We should modify scheduler to always have that attribute.
                if hasattr(self.scheduler, "is_complete") and self.scheduler.is_complete:  # type: ignore
                    logger.info("[SweepController]  Sweep completed successfully.")
                    break

                # Filter jobs based on capacity constraints
                # EVAL jobs always go through, TRAINING jobs respect the limit
                filtered_jobs = []
                training_jobs_count = 0

                for job in new_jobs:
                    if job.type == JobTypes.LAUNCH_EVAL:
                        # Always allow eval jobs
                        filtered_jobs.append(job)
                    elif job.type == JobTypes.LAUNCH_TRAINING:
                        # Check if we have capacity for training jobs
                        if metadata.runs_in_progress + training_jobs_count < self.max_parallel_jobs:
                            filtered_jobs.append(job)
                            training_jobs_count += 1
                        else:
                            logger.debug(
                                f"[SweepController] At max parallel jobs limit ({self.max_parallel_jobs}), "
                                f"skipping training job {job.run_id}"
                            )
                    else:
                        # Other job types (if any) go through
                        filtered_jobs.append(job)

                new_jobs = filtered_jobs

                # 4. Execute scheduler's decisions
                for job in new_jobs:
                    try:
                        if job.type == JobTypes.LAUNCH_TRAINING:
                            self.store.init_run(job.run_id, sweep_id=self.sweep_id)
                            # Store the suggestion (hyperparameters) in the run summary
                            if job.config:  # job.config contains the optimizer suggestion
                                self.store.update_run_summary(job.run_id, {"suggestion": job.config})
                            logger.info(f"[SweepController] Created run {job.run_id}")
                        elif job.type == JobTypes.LAUNCH_EVAL:
                            success = self.store.update_run_summary(
                                job.run_id,
                                {
                                    "has_started_eval": True,  # other status properties can be deduced from this.
                                },
                            )
                            if not success:
                                logger.error(
                                    f"[SweepController] Failed to update run summary for eval job {job.run_id}, "
                                    "skipping dispatch"
                                )
                                raise RuntimeError(f"Failed to update run summary for eval job {job.run_id}")
                            # Small delay to allow W&B API to propagate the flag update
                            time.sleep(1)
                            logger.info(f"[SweepController] Launching eval for job {job.run_id}")

                        # Only dispatch if store operations succeeded
                        dispatch_id = self.dispatcher.dispatch(job)
                        logger.info(f"[SweepController] Dispatched {job.run_id} with dispatch_id {dispatch_id}")

                    except Exception as e:
                        logger.error(f"[SweepController] Failed to initialize/dispatch job {job.run_id}: {e}")
                        logger.error(
                            f"[SweepController] Skipping dispatch for {job.run_id} to prevent resource overload"
                        )
                        # Continue with next job rather than crashing the whole sweep
                        continue

                # 5. Finally, update transient states and mark completions
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

                # 5. Sleep
                time.sleep(self.monitoring_interval)

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
