"""Simplified adaptive experiment controller."""

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from metta.common.util.retry import retry_function

from .adaptive_config import AdaptiveConfig
from .models import JobDefinition, JobStatus, JobTypes, RunInfo
from .protocols import (
    Dispatcher,
    ExperimentScheduler,
    Store,
)
from .utils import make_monitor_table

logger = logging.getLogger(__name__)


class AdaptiveController:
    """
    Simple controller for adaptive experiments.

    Everything is inlined in the main run() method for maximum clarity.
    """

    def __init__(
        self,
        experiment_id: str,
        scheduler: ExperimentScheduler,
        dispatcher: Dispatcher,
        store: Store,
        config: AdaptiveConfig,
    ):
        self.experiment_id = experiment_id
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.store = store
        self.config = config

        # Job tracking by (run_id, job_type) to handle train/eval jobs with same run_id
        self.dispatched_jobs: set[tuple[str, str]] = set()

    def run(
        self,
        on_training_completed: Optional[Callable[[RunInfo, Store, list[RunInfo]], None]] = None,
        on_eval_completed: Optional[Callable[[RunInfo, Store, list[RunInfo]], None]] = None,
        on_job_dispatch: Optional[Callable[[JobDefinition, Store], None]] = None,
    ) -> None:
        """Main adaptive experiment loop - everything inline."""
        logger.info(f"[AdaptiveController] Starting experiment {self.experiment_id}")
        has_data = self.config.resume

        while True:
            try:
                # 1. Get current state
                if has_data:
                    time.sleep(self.config.monitoring_interval)
                    try:
                        runs = self.store.fetch_runs(filters={"group": self.experiment_id})
                    except Exception as e:
                        logger.error("Error when fetching WandB runs")
                        raise e
                else:
                    runs = []
                    has_data = True  # Skip first fetch because WandB will just timeout.

                # Display monitoring table every interval
                if runs:
                    table_lines = make_monitor_table(
                        runs=runs,
                        title="Run Status Table",
                        logger_prefix="[AdaptiveController]",
                        include_score=True,
                        truncate_run_id=True,
                    )
                    for line in table_lines:
                        logger.info(line)

                # 1.a Run post-eval completion hooks (guarded by summary flag) before any scheduling
                if runs and on_training_completed is not None:
                    for run in runs:
                        try:
                            summary_dict = run.summary or {}
                            already_processed = bool(summary_dict.get("adaptive/post_train_processed", False))
                            if run.has_completed_training and not already_processed:
                                logger.info(f"[AdaptiveController] Running on_training_completed for {run.run_id}")
                                on_training_completed(run, self.store, runs)
                                processed_at = datetime.now(timezone.utc)
                                self.store.update_run_summary(
                                    run.run_id,
                                    {
                                        "adaptive/post_train_processed": True,
                                        "adaptive/post_train_processed_at": processed_at,
                                    },
                                )
                        except Exception as e:
                            logger.error(
                                f"[AdaptiveController] Error running on_training_completed for {run.run_id}: {e}"
                            )

                if runs and on_eval_completed is not None:
                    for run in runs:
                        try:
                            summary_dict = run.summary or {}
                            already_processed = bool(summary_dict.get("adaptive/post_eval_processed", False))
                            if run.has_been_evaluated and not already_processed:
                                logger.info(f"[AdaptiveController] Running on_eval_completed for {run.run_id}")

                                retry_function(
                                    lambda r=run, rs=runs: on_eval_completed(r, self.store, rs),  # type: ignore
                                    max_retries=3,
                                    initial_delay=1.0,
                                    max_delay=30.0,
                                )

                                processed_at = datetime.now(timezone.utc)
                                self.store.update_run_summary(
                                    run.run_id,
                                    {
                                        "adaptive/post_eval_processed": True,
                                        "adaptive/post_eval_processed_at": processed_at,
                                    },
                                )
                        except Exception as e:
                            logger.error(f"[AdaptiveController] on_eval_completed failed for {run.run_id}: {e}")

                # 2. Calculate available training slots (only count runs actually using training resources)
                active_training_count = sum(
                    1 for run in runs if run.status in (JobStatus.PENDING, JobStatus.IN_TRAINING)
                )
                available_training_slots = max(0, self.config.max_parallel - active_training_count)

                # 3. Let scheduler decide (with resource awareness)
                # This also updates internal state for completed runs
                new_jobs = self.scheduler.schedule(runs, available_training_slots)

                # 4. Check if scheduler says experiment is complete (after state updates)
                if self.scheduler.is_experiment_complete(runs):
                    logger.info("[AdaptiveController] Scheduler reports experiment complete")
                    break

                if not new_jobs:
                    # No new jobs, wait before next check
                    continue

                # 5. Validate training job constraint
                training_jobs = [j for j in new_jobs if j.type == JobTypes.LAUNCH_TRAINING]
                if len(training_jobs) > available_training_slots:
                    logger.error(
                        f"[AdaptiveController] Scheduler requested {len(training_jobs)} training jobs "
                        f"but only {available_training_slots} slots available. Skipping cycle."
                    )
                    continue

                # 6. Dispatch all jobs
                for job in new_jobs:
                    # Create job key for tracking
                    job_key = (job.run_id, job.type.value)

                    # Skip if already dispatched
                    if job_key in self.dispatched_jobs:
                        logger.debug(f"[AdaptiveController] Job {job.run_id} ({job.type}) already dispatched")
                        continue

                    try:
                        # Dispatch the job
                        dispatch_id = self.dispatcher.dispatch(job)
                        self.dispatched_jobs.add(job_key)

                        # Initialize run in store (only for training jobs, eval reuses same run)
                        if job.type == JobTypes.LAUNCH_TRAINING:
                            # Pass job metadata as initial summary data
                            self.store.init_run(job.run_id, group=self.experiment_id, initial_summary=job.metadata)

                        # Mark eval jobs as started in store
                        elif job.type == JobTypes.LAUNCH_EVAL:
                            self.store.update_run_summary(job.run_id, {"has_started_eval": True})

                        # Call job dispatch hook if provided (after wandb initialization)
                        if on_job_dispatch is not None:
                            try:
                                on_job_dispatch(job, self.store)
                            except Exception as e:
                                logger.error(f"[AdaptiveController] on_job_dispatch failed for {job.run_id}: {e}")

                        logger.info(
                            f"[AdaptiveController] Dispatched {job.run_id} ({job.type}) (dispatch_id: {dispatch_id})"
                        )

                    except Exception as e:
                        logger.error(f"[AdaptiveController] Failed to dispatch {job.run_id} ({job.type}): {e}")

            except KeyboardInterrupt:
                logger.info("[AdaptiveController] Interrupted, stopping experiment")
                break
            except Exception as e:
                logger.error(f"[AdaptiveController] Error in control loop: {e}")

        logger.info(f"[AdaptiveController] Experiment {self.experiment_id} complete")
