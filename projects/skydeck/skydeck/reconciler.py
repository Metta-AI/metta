"""Reconciliation engine to align current state with desired state."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .database import Database
from .desired_state import DesiredStateManager
from .models import DesiredState, Experiment, Job, JobStatus
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class Reconciler:
    """Reconciliation engine that aligns current state with desired state.

    Periodically checks experiments and takes actions to bring current_state
    in line with desired_state:
    - LAUNCH: Create new job if desired=RUNNING and current=INIT/STOPPED/FAILED
    - START: Start stopped cluster if desired=RUNNING
    - STOP: Stop cluster if desired=STOPPED
    - TERMINATE: Terminate cluster if desired=TERMINATED
    - CANCEL_JOB: Cancel running job if desired=STOPPED/TERMINATED
    """

    def __init__(
        self,
        db: Database,
        desired_state_manager: DesiredStateManager,
        state_manager: StateManager,
        interval: int = 60,
    ):
        """Initialize reconciler.

        Args:
            db: Database instance
            desired_state_manager: DesiredStateManager instance
            state_manager: StateManager instance
            interval: Reconciliation interval in seconds (default: 60)
        """
        self.db = db
        self.desired_state_manager = desired_state_manager
        self.state_manager = state_manager
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.last_reconcile: Optional[datetime] = None

    async def start(self):
        """Start the background reconciliation task."""
        if self.running:
            logger.warning("Reconciler already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._reconcile_loop())
        logger.info(f"Reconciler started (interval={self.interval}s)")

    async def stop(self):
        """Stop the background reconciliation task."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Reconciler stopped")

    async def reconcile_once(self):
        """Execute one reconciliation cycle immediately."""
        try:
            await self._reconcile()
            self.last_reconcile = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error during reconciliation: {e}", exc_info=True)

    async def _reconcile_loop(self):
        """Main reconciliation loop."""
        while self.running:
            try:
                await self._reconcile()
                self.last_reconcile = datetime.utcnow()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during reconciliation: {e}", exc_info=True)

            # Wait for next interval
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def _reconcile(self):
        """Execute one reconciliation cycle.

        Checks all experiments and takes actions to align current with desired state.
        """
        logger.debug("Starting reconciliation cycle...")

        experiments = await self.desired_state_manager.get_experiments_needing_reconciliation()
        if not experiments:
            logger.debug("No experiments need reconciliation")
            return

        logger.info(f"Reconciling {len(experiments)} experiments")

        for experiment in experiments:
            try:
                await self._reconcile_experiment(experiment)
            except Exception as e:
                logger.error(f"Error reconciling experiment {experiment.id}: {e}", exc_info=True)

    async def _reconcile_experiment(self, experiment: Experiment):
        """Reconcile a single experiment.

        Args:
            experiment: Experiment to reconcile
        """
        logger.info(
            f"Reconciling {experiment.id}: desired={experiment.desired_state}, current={experiment.current_state}"
        )

        if experiment.desired_state == DesiredState.RUNNING:
            await self._reconcile_to_running(experiment)
        elif experiment.desired_state == DesiredState.STOPPED:
            await self._reconcile_to_stopped(experiment)
        elif experiment.desired_state == DesiredState.TERMINATED:
            await self._reconcile_to_terminated(experiment)

    async def _reconcile_to_running(self, experiment: Experiment):
        """Reconcile experiment to RUNNING state.

        Args:
            experiment: Experiment to reconcile
        """
        # If already running or pending, nothing to do
        if experiment.current_state in {JobStatus.RUNNING, JobStatus.PENDING}:
            logger.debug(f"Experiment {experiment.id} already running/pending")
            return

        # Check if there's an active job
        if experiment.current_job_id:
            job = await self.db.get_job(experiment.current_job_id)
            if job and job.is_active():
                logger.debug(f"Experiment {experiment.id} has active job {job.id}")
                return

        # Need to launch new job
        await self._launch_job(experiment)

    async def _reconcile_to_stopped(self, experiment: Experiment):
        """Reconcile experiment to STOPPED state.

        Args:
            experiment: Experiment to reconcile
        """
        # If there's a running job, cancel it
        if experiment.current_job_id:
            job = await self.db.get_job(experiment.current_job_id)
            if job and job.is_active():
                await self._cancel_job(job)

        # If cluster is running, stop it
        if experiment.cluster_name:
            cluster = await self.state_manager.get_cluster_status(experiment.cluster_name)
            if cluster and cluster.status == "UP":
                await self._stop_cluster(experiment.cluster_name)

    async def _reconcile_to_terminated(self, experiment: Experiment):
        """Reconcile experiment to TERMINATED state.

        Args:
            experiment: Experiment to reconcile
        """
        # Cancel any running job
        if experiment.current_job_id:
            job = await self.db.get_job(experiment.current_job_id)
            if job and job.is_active():
                await self._cancel_job(job)

        # Terminate cluster
        if experiment.cluster_name:
            await self._terminate_cluster(experiment.cluster_name)

        # Update experiment state
        await self.db.update_experiment_state(experiment.id, current_state=JobStatus.INIT, current_job_id=None)

    async def _launch_job(self, experiment: Experiment):
        """Launch a new job for an experiment.

        Args:
            experiment: Experiment to launch
        """
        try:
            # Generate cluster name if not set
            if not experiment.cluster_name:
                experiment.cluster_name = f"{experiment.id}-cluster"
                await self.db.set_experiment_cluster(experiment.id, experiment.cluster_name)

            # Build command
            command = experiment.build_command()

            # Create job record
            # Get next job number for this experiment
            existing_jobs = await self.db.get_jobs_for_experiment(experiment.id, limit=1)
            job_number = 1
            if existing_jobs:
                # Extract job number from last job ID
                last_id = existing_jobs[0].id
                try:
                    last_number = int(last_id.split("-")[-1])
                    job_number = last_number + 1
                except (ValueError, IndexError):
                    pass

            job = Job(
                id=f"{experiment.id}-{job_number}",
                experiment_id=experiment.id,
                cluster_name=experiment.cluster_name,
                status=JobStatus.PENDING,
                command=command,
                nodes=experiment.nodes,
                gpus=experiment.gpus,
                instance_type=experiment.instance_type,
                cloud=experiment.cloud,
                region=experiment.region,
                zone=experiment.zone,
                submitted_at=datetime.utcnow(),
            )

            try:
                await self.db.save_job(job, allow_update=False)
            except ValueError as e:
                # Job ID collision - this shouldn't happen but could in rare race conditions
                logger.error(f"Failed to create job {job.id}: {e}")
                logger.info("Retrying with incremented job number...")
                # Retry with next job number
                job.id = f"{experiment.id}-{job_number + 1}"
                await self.db.save_job(job, allow_update=False)
            await self.db.update_experiment_state(experiment.id, current_state=JobStatus.PENDING, current_job_id=job.id)

            logger.info(f"Created job {job.id} for experiment {experiment.id}")

            # Launch on SkyPilot
            sky_job_id = await self._sky_launch(
                cluster_name=experiment.cluster_name,
                command=command,
                nodes=experiment.nodes,
                gpus=experiment.gpus,
                instance_type=experiment.instance_type,
                cloud=experiment.cloud,
                region=experiment.region,
                zone=experiment.zone,
                spot=experiment.spot,
            )

            # Update job with SkyPilot job ID
            if sky_job_id is not None:
                job.sky_job_id = sky_job_id
                await self.db.save_job(job)
                logger.info(f"Launched job {job.id} on SkyPilot (sky_job_id={sky_job_id})")

        except Exception as e:
            logger.error(f"Failed to launch job for {experiment.id}: {e}", exc_info=True)
            # Update experiment state to FAILED
            await self.db.update_experiment_state(experiment.id, current_state=JobStatus.FAILED)

    async def _cancel_job(self, job: Job):
        """Cancel a running job.

        Args:
            job: Job to cancel
        """
        try:
            logger.info(f"Cancelling job {job.id}")

            # Cancel on SkyPilot
            if job.sky_job_id is not None:
                await self._sky_cancel(job.cluster_name, job.sky_job_id)

            # Update job status
            await self.db.update_job_status(
                job.id,
                status=JobStatus.CANCELLED,
                ended_at=datetime.utcnow(),
            )

            # Update experiment state
            await self.db.update_experiment_state(
                job.experiment_id, current_state=JobStatus.CANCELLED, current_job_id=None
            )

        except Exception as e:
            logger.error(f"Failed to cancel job {job.id}: {e}", exc_info=True)

    async def _stop_cluster(self, cluster_name: str):
        """Stop a cluster.

        Args:
            cluster_name: Name of cluster to stop
        """
        try:
            logger.info(f"Stopping cluster {cluster_name}")
            await self._sky_stop(cluster_name)
        except Exception as e:
            logger.error(f"Failed to stop cluster {cluster_name}: {e}", exc_info=True)

    async def _terminate_cluster(self, cluster_name: str):
        """Terminate a cluster.

        Args:
            cluster_name: Name of cluster to terminate
        """
        try:
            logger.info(f"Terminating cluster {cluster_name}")
            await self._sky_down(cluster_name)
        except Exception as e:
            logger.error(f"Failed to terminate cluster {cluster_name}: {e}", exc_info=True)

    # SkyPilot integration methods

    async def _sky_launch(
        self,
        cluster_name: str,
        command: str,
        nodes: int = 1,
        gpus: int = 0,
        instance_type: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        spot: bool = False,
    ) -> Optional[int]:
        """Launch job on SkyPilot.

        Args:
            cluster_name: Cluster name
            command: Command to run
            nodes: Number of nodes
            gpus: GPUs per node
            instance_type: Instance type
            cloud: Cloud provider
            region: Region
            zone: Zone
            spot: Use spot instances

        Returns:
            SkyPilot job ID or None
        """
        try:
            import sky

            # Build SkyPilot task
            # This is a simplified example - you'll need to adapt to actual SkyPilot API
            task = sky.Task(run=command)

            # Set resources
            resources = sky.Resources()
            if cloud:
                resources = resources.cloud(getattr(sky.clouds, cloud.upper(), None))
            if instance_type:
                resources = resources.instance_type(instance_type)
            if gpus:
                resources = resources.accelerators(f"V100:{gpus}")  # Adjust as needed
            if region:
                resources = resources.region(region)
            if zone:
                resources = resources.zone(zone)
            if spot:
                resources = resources.use_spot(True)

            task.set_resources(resources)

            # Launch
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                sky.launch,
                task,
                cluster_name,
                False,  # detach=False
            )

            # Extract job ID from result (format depends on SkyPilot version)
            # This is a placeholder - adjust based on actual return value
            return result if isinstance(result, int) else None

        except ImportError:
            logger.error("SkyPilot not installed")
            return None
        except Exception as e:
            logger.error(f"Error launching on SkyPilot: {e}", exc_info=True)
            return None

    async def _sky_cancel(self, cluster_name: str, job_id: int):
        """Cancel job on SkyPilot.

        Args:
            cluster_name: Cluster name
            job_id: SkyPilot job ID
        """
        try:
            import sky

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sky.cancel, cluster_name, job_id)
        except Exception as e:
            logger.error(f"Error cancelling job on SkyPilot: {e}", exc_info=True)
            raise

    async def _sky_stop(self, cluster_name: str):
        """Stop cluster on SkyPilot.

        Args:
            cluster_name: Cluster name
        """
        try:
            import sky

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sky.stop, cluster_name)
        except Exception as e:
            logger.error(f"Error stopping cluster on SkyPilot: {e}", exc_info=True)
            raise

    async def _sky_down(self, cluster_name: str):
        """Terminate cluster on SkyPilot.

        Args:
            cluster_name: Cluster name
        """
        try:
            import sky

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sky.down, cluster_name)
        except Exception as e:
            logger.error(f"Error terminating cluster on SkyPilot: {e}", exc_info=True)
            raise
