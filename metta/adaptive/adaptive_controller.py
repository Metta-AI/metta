"""Simplified adaptive experiment controller."""

import logging
import time
from datetime import datetime
from typing import Callable, Optional

from metta.common.util.retry import retry_function

from .adaptive_config import AdaptiveConfig
from .models import JobStatus, JobTypes, RunInfo
from .protocols import (
    Dispatcher,
    ExperimentScheduler,
    SchedulerWithState,
    StateStore,
    Store,
)

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
        on_eval_completed: Optional[Callable[[RunInfo, Store, list[RunInfo]], None]] = None,
        state_store: Optional[StateStore] = None,
    ):
        self.experiment_id = experiment_id
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.store = store
        self.config = config
        self.on_eval_completed = on_eval_completed
        self.state_store = state_store

        # Job tracking by (run_id, job_type) to handle train/eval jobs with same run_id
        self.dispatched_jobs: set[tuple[str, str]] = set()

    def run(self) -> None:
        """Main adaptive experiment loop - everything inline."""
        logger.info(f"[AdaptiveController] Starting experiment {self.experiment_id}")
        has_data = self.config.resume

        loaded_state = False
        while True:
            try:
                # 1. Get current state
                if has_data:
                    time.sleep(self.config.monitoring_interval)
                    runs = self.store.fetch_runs(filters={"group": self.experiment_id})
                else:
                    runs = []
                    has_data = True  # Skip first fetch because WandB will just timeout.

                # 1.0 Load scheduler state on first data fetch if supported
                if (
                    not loaded_state
                    and self.state_store is not None
                    and isinstance(self.scheduler, SchedulerWithState)
                ):
                    try:
                        if self.scheduler.should_load_from_store(runs):  # type: ignore[attr-defined]
                            self.scheduler.load_from_store(self.state_store, self.experiment_id)  # type: ignore[attr-defined]
                            logger.info("[AdaptiveController] Loaded scheduler state from store")
                    except Exception as e:
                        logger.warning(f"[AdaptiveController] Failed to load scheduler state: {e}")
                    finally:
                        loaded_state = True

                # 1.a Run post-eval completion hooks (guarded by summary flag) before any scheduling
                if runs and self.on_eval_completed is not None:
                    for run in runs:
                        try:
                            summary_dict = run.summary if isinstance(run.summary, dict) else {}
                            already_processed = bool(
                                summary_dict.get("adaptive/post_eval_processed", False)
                            )
                            if run.has_been_evaluated and not already_processed:
                                logger.info(
                                    f"[AdaptiveController] Running on_eval_completed for {run.run_id}"
                                )

                                def _invoke(r: RunInfo = run, rs: list[RunInfo] = runs) -> None:
                                    assert self.on_eval_completed is not None
                                    self.on_eval_completed(r, self.store, rs)

                                retry_function(_invoke, max_retries=3, initial_delay=1.0, max_delay=30.0)

                                processed_at = datetime.utcnow().isoformat()
                                self.store.update_run_summary(
                                    run.run_id,
                                    {
                                        "adaptive/post_eval_processed": True,
                                        "adaptive/post_eval_processed_at": processed_at,
                                    },
                                )

                                if isinstance(run.summary, dict):
                                    run.summary["adaptive/post_eval_processed"] = True
                                    run.summary["adaptive/post_eval_processed_at"] = processed_at
                        except Exception as e:
                            logger.error(
                                f"[AdaptiveController] on_eval_completed failed for {run.run_id}: {e}"
                            )

                # 1.b Save scheduler state after processing eval completions (if supported)
                if self.state_store is not None and isinstance(self.scheduler, SchedulerWithState):
                    try:
                        self.scheduler.save_to_store(self.state_store, self.experiment_id)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning(f"[AdaptiveController] Failed to save scheduler state: {e}")

                # 2. Check if scheduler says experiment is complete
                if self.scheduler.is_experiment_complete(runs):
                    logger.info("[AdaptiveController] Scheduler reports experiment complete")
                    break

                # 3. Calculate available training slots (only count runs actually using training resources)
                active_training_count = sum(
                    1 for run in runs if run.status in (JobStatus.PENDING, JobStatus.IN_TRAINING)
                )
                available_training_slots = max(0, self.config.max_parallel - active_training_count)

                # 4. Let scheduler decide (with resource awareness)
                new_jobs = self.scheduler.schedule(runs, available_training_slots)

                if not new_jobs:
                    # No new jobs, but check if experiment is complete before waiting
                    if self.scheduler.is_experiment_complete(runs):
                        logger.info("[AdaptiveController] No new jobs and scheduler reports experiment complete")
                        break
                    time.sleep(self.config.monitoring_interval)
                    continue

                # 5. Separate by job type and validate
                training_jobs = [j for j in new_jobs if j.type == JobTypes.LAUNCH_TRAINING]
                eval_jobs = [j for j in new_jobs if j.type == JobTypes.LAUNCH_EVAL]

                # 6. Validate training job constraint
                if len(training_jobs) > available_training_slots:
                    logger.error(
                        f"[AdaptiveController] Scheduler requested {len(training_jobs)} training jobs "
                        f"but only {available_training_slots} slots available. Skipping cycle."
                    )
                    continue

                # 7. Dispatch all jobs
                all_jobs = training_jobs + eval_jobs
                for job in all_jobs:
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
                            self.store.init_run(job.run_id, group=self.experiment_id)
                            # If scheduler attached a suggestion, persist it to summary for later hooks/optimizers
                            suggestion = (
                                job.metadata.get("adaptive/suggestion") if isinstance(job.metadata, dict) else None
                            )
                            if suggestion is not None:
                                self.store.update_run_summary(job.run_id, {"observation/suggestion": suggestion})

                        # Mark eval jobs as started in store
                        elif job.type == JobTypes.LAUNCH_EVAL:
                            self.store.update_run_summary(job.run_id, {"has_started_eval": True})

                        logger.info(
                            f"[AdaptiveController] Dispatched {job.run_id} ({job.type}) (dispatch_id: {dispatch_id})"
                        )

                    except Exception as e:
                        logger.error(f"[AdaptiveController] Failed to dispatch {job.run_id} ({job.type}): {e}")

                # 8. Save scheduler state after scheduling (if supported)
                if self.state_store is not None and isinstance(self.scheduler, SchedulerWithState):
                    try:
                        self.scheduler.save_to_store(self.state_store, self.experiment_id)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.warning(f"[AdaptiveController] Failed to save scheduler state: {e}")

            except KeyboardInterrupt:
                logger.info("[AdaptiveController] Interrupted, stopping experiment")
                break
            except Exception as e:
                logger.error(f"[AdaptiveController] Error in control loop: {e}")
                time.sleep(self.config.monitoring_interval)

        logger.info(f"[AdaptiveController] Experiment {self.experiment_id} complete")
