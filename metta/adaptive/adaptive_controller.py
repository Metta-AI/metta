"""Simplified adaptive experiment controller."""

import logging
import time

from .adaptive_config import AdaptiveConfig
from .protocols import Dispatcher, ExperimentScheduler, Store

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

        # Simple job tracking
        self.dispatched_jobs = set[str]()

    def run(self) -> None:
        """Main adaptive experiment loop - everything inline."""
        logger.info(f"[AdaptiveController] Starting experiment {self.experiment_id}")
        has_data = self.config.resume
        while len(self.dispatched_jobs) < self.config.max_trials:
            try:
                # 1. Get current state
                if has_data:
                    time.sleep(self.config.monitoring_interval)
                    runs = self.store.fetch_runs(filters={"group": self.experiment_id})
                else:
                    runs = []
                    has_data = True # Skip first fetch because WandB will just timeout.

                # 2. Calculate available training slots
                active_training_count = sum(
                    1 for run in runs
                    if not (hasattr(run, 'status') and run.status.value in ('completed', 'failed'))
                )
                available_training_slots = max(0, self.config.max_parallel - active_training_count)

                # 3. Let scheduler decide (with resource awareness)
                new_jobs = self.scheduler.schedule(runs, available_training_slots)

                if not new_jobs:
                    time.sleep(self.config.monitoring_interval)
                    continue

                # 4. Separate by job type and validate
                from .models import JobTypes
                training_jobs = [j for j in new_jobs if j.type == JobTypes.LAUNCH_TRAINING]
                eval_jobs = [j for j in new_jobs if j.type == JobTypes.LAUNCH_EVAL]

                # 5. Validate training job constraint
                if len(training_jobs) > available_training_slots:
                    logger.error(
                        f"[AdaptiveController] Scheduler requested {len(training_jobs)} training jobs "
                        f"but only {available_training_slots} slots available. Skipping cycle."
                    )
                    continue

                # 6. Dispatch all jobs
                all_jobs = training_jobs + eval_jobs
                for job in all_jobs:
                    # Skip if already dispatched
                    if job.run_id in self.dispatched_jobs:
                        logger.debug(f"[AdaptiveController] Job {job.run_id} already dispatched")
                        continue

                    try:
                        # Dispatch the job
                        dispatch_id = self.dispatcher.dispatch(job)
                        self.dispatched_jobs.add(job.run_id)

                        # Initialize run in store
                        self.store.init_run(job.run_id, group=self.experiment_id)

                        # Store job config
                        if job.config:
                            self.store.update_run_summary(job.run_id, {"config": job.config})
                        logger.info(f"[AdaptiveController] Dispatched {job.run_id} (dispatch_id: {dispatch_id})")

                    except Exception as e:
                        logger.error(f"[AdaptiveController] Failed to dispatch {job.run_id}: {e}")

            except KeyboardInterrupt:
                logger.info("[AdaptiveController] Interrupted, stopping experiment")
                break
            except Exception as e:
                logger.error(f"[AdaptiveController] Error in control loop: {e}")
                time.sleep(self.config.monitoring_interval)

        logger.info(f"[AdaptiveController] Experiment {self.experiment_id} complete")
