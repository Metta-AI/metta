"""State manager for caching current SkyPilot state."""

import logging
from datetime import datetime
from typing import Optional

from .database import Database
from .models import Cluster, Job, JobStatus

logger = logging.getLogger(__name__)


def map_job_status_to_experiment_state(job_status: JobStatus) -> JobStatus:
    """Map job status to simplified experiment state.

    Experiment states should be: Running, Stopped, Failed, Done
    - RUNNING, PENDING, STARTING → Running (or Starting if STARTING)
    - FAILED → Failed
    - CANCELLED → Stopped
    - SUCCEEDED → Done
    - INIT → Stopped
    """
    if job_status == JobStatus.STARTING:
        return JobStatus.STARTING  # Keep STARTING as-is to show yellow "Starting"
    elif job_status in {JobStatus.RUNNING, JobStatus.PENDING}:
        return JobStatus.RUNNING
    elif job_status == JobStatus.FAILED:
        return JobStatus.FAILED
    elif job_status == JobStatus.CANCELLED:
        return JobStatus.INIT  # Using INIT to represent "Stopped"
    elif job_status == JobStatus.SUCCEEDED:
        return JobStatus.SUCCEEDED  # Will display as "Done"
    else:  # INIT, UNKNOWN, etc.
        return JobStatus.INIT


class StateManager:
    """Manages cached state from SkyPilot.

    This provides a read-only view of the actual infrastructure state
    by caching information from sky.status() and sky.queue() calls.
    """

    def __init__(self, db: Database):
        """Initialize state manager.

        Args:
            db: Database instance for persistence
        """
        self.db = db

    async def update_cluster_state(self, sky_clusters: list[dict]):
        """Update cluster state from SkyPilot status.

        Args:
            sky_clusters: List of cluster dicts from sky.status()
        """
        for sky_cluster in sky_clusters:
            cluster = self._parse_sky_cluster(sky_cluster)
            await self.db.save_cluster(cluster)
            logger.debug(f"Updated cluster state: {cluster.name} ({cluster.status})")

    async def update_job_state(self, cluster_name: str, sky_jobs: list[dict]):
        """Update job state from SkyPilot queue.

        Args:
            cluster_name: Name of the cluster
            sky_jobs: List of job dicts from sky.queue(cluster_name)
        """
        for sky_job in sky_jobs:
            job_id = sky_job.get("job_id")
            if not job_id:
                continue

            # Try to find matching job in database
            # This is tricky - we need to match SkyPilot jobs to our Job records
            # For now, we'll update based on sky_job_id if we have it
            job = await self._find_job_by_sky_id(cluster_name, job_id)
            if job:
                # Update existing job status
                status = self._parse_sky_job_status(sky_job.get("status", "UNKNOWN"))
                started_at = job.started_at
                ended_at = job.ended_at

                # Update timestamps based on status
                if status == JobStatus.RUNNING and not started_at:
                    started_at = datetime.utcnow()
                elif status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                    if not ended_at:
                        ended_at = datetime.utcnow()

                await self.db.update_job_status(
                    job.id,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                logger.debug(f"Updated job state: {job.id} -> {status}")

                # Update experiment current state (job.experiment_id is the experiment name)
                experiment = await self.db.get_experiment_by_name(job.experiment_id)
                if experiment:
                    # Only update if job is actively running
                    if status in {JobStatus.RUNNING, JobStatus.PENDING}:
                        mapped_state = map_job_status_to_experiment_state(status)
                        await self.db.update_experiment_state(experiment.id, current_state=mapped_state)

    async def get_cluster_status(self, cluster_name: str) -> Optional[Cluster]:
        """Get current status of a cluster.

        Args:
            cluster_name: Name of the cluster

        Returns:
            Cluster object or None if not found
        """
        return await self.db.get_cluster(cluster_name)

    async def get_all_clusters(self) -> list[Cluster]:
        """Get all known clusters.

        Returns:
            List of Cluster objects
        """
        return await self.db.get_all_clusters()

    async def update_managed_jobs(self, sky_jobs: list[dict]):
        """Update managed jobs from SkyPilot.

        Args:
            sky_jobs: List of job dicts from sky.jobs.queue()
        """
        from datetime import datetime

        # Build set of active SkyPilot job IDs for orphan detection
        sky_job_ids = {sky_job.get("id") for sky_job in sky_jobs if sky_job.get("id")}

        # Get all active jobs from database
        active_jobs = await self.db.get_running_jobs()

        # Mark any jobs that are active in database but not in SkyPilot as FAILED (orphaned)
        for active_job in active_jobs:
            if active_job.id not in sky_job_ids:
                logger.info(f"Marking orphaned job {active_job.id} as FAILED (not found in SkyPilot)")
                await self.db.update_job_status(
                    active_job.id,
                    status=JobStatus.FAILED,
                    started_at=active_job.started_at,
                    ended_at=datetime.utcnow(),
                )

                # Update experiment state (job.experiment_id is the experiment name)
                experiment = await self.db.get_experiment_by_name(active_job.experiment_id)
                if experiment:
                    mapped_state = map_job_status_to_experiment_state(JobStatus.FAILED)
                    await self.db.update_experiment_state(
                        experiment.id, current_state=mapped_state
                    )
                    logger.info(f"Updated experiment {experiment.id} current_state to {mapped_state}")

        # Track which experiments we've seen active jobs for
        experiments_with_active_jobs = set()

        # Update jobs that exist in both SkyPilot and database
        for sky_job in sky_jobs:
            job_id = sky_job.get("id")
            if not job_id:
                continue

            # Check if job already exists in database
            existing_job = await self.db.get_job(job_id)

            if existing_job:
                # Update existing job
                status_str = sky_job.get("status", "UNKNOWN")
                try:
                    status = JobStatus[status_str.upper()]
                except (KeyError, AttributeError):
                    status = JobStatus.UNKNOWN

                await self.db.update_job_status(
                    job_id,
                    status=status,
                    started_at=existing_job.started_at,  # Don't override timestamps
                    ended_at=existing_job.ended_at,
                )

                # Update experiment current state (job.experiment_id is the experiment name)
                # Only update for ACTIVE jobs (RUNNING/PENDING)
                experiment = await self.db.get_experiment_by_name(existing_job.experiment_id)
                if experiment and status in {JobStatus.RUNNING, JobStatus.PENDING}:
                    mapped_state = map_job_status_to_experiment_state(status)
                    await self.db.update_experiment_state(experiment.id, current_state=mapped_state)
                    logger.info(f"Updated experiment {experiment.name} (id={experiment.id}) current_state to {mapped_state} from job {job_id}")
                    experiments_with_active_jobs.add(existing_job.experiment_id)
            else:
                # Create new job record
                # For managed jobs, we don't have an experiment_id from our system
                # We'll use the job name as both experiment_id and job_id
                status_str = sky_job.get("status", "UNKNOWN")
                try:
                    status = JobStatus[status_str.upper()]
                except (KeyError, AttributeError):
                    status = JobStatus.UNKNOWN

                # Create a Job object
                job = Job(
                    id=job_id,
                    experiment_id=sky_job.get("name", job_id),  # Use job name as experiment
                    cluster_name=sky_job.get("infra", "managed"),
                    sky_job_id=int(job_id) if job_id.isdigit() else None,
                    command=sky_job.get("entrypoint", ""),
                    nodes=sky_job.get("nodes", 1),
                    gpus=sky_job.get("gpus", 0),
                    status=status,
                    created_at=(
                        datetime.fromtimestamp(sky_job.get("submitted_at", 0))
                        if sky_job.get("submitted_at")
                        else datetime.utcnow()
                    ),
                    started_at=(
                        datetime.fromtimestamp(sky_job.get("start_at", 0)) if sky_job.get("start_at") else None
                    ),
                    ended_at=(datetime.fromtimestamp(sky_job.get("end_at", 0)) if sky_job.get("end_at") else None),
                    exit_code=None,
                )

                await self.db.save_job(job)
                logger.debug(f"Created new managed job: {job_id} ({status})")

        # For experiments with no active jobs, update state based on most recent terminal job
        all_experiments = await self.db.get_all_experiments()
        for experiment in all_experiments:
            if experiment.name not in experiments_with_active_jobs:
                # Get the most recent job for this experiment
                recent_jobs = await self.db.get_jobs_for_experiment(experiment.name, limit=1)
                if recent_jobs:
                    most_recent_job = recent_jobs[0]
                    # Only update if the job is in a terminal state
                    if most_recent_job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                        mapped_state = map_job_status_to_experiment_state(most_recent_job.status)
                        if experiment.current_state != mapped_state:
                            await self.db.update_experiment_state(experiment.id, current_state=mapped_state)
                            logger.info(f"Updated experiment {experiment.name} (id={experiment.id}) current_state to {mapped_state} (no active jobs, using most recent job {most_recent_job.id})")

    async def _find_job_by_sky_id(self, cluster_name: str, sky_job_id: int) -> Optional[Job]:
        """Find job by SkyPilot job ID.

        Args:
            cluster_name: Cluster name
            sky_job_id: SkyPilot's internal job ID

        Returns:
            Job object or None
        """
        # Query database for job with matching sky_job_id and cluster
        cursor = await self.db._conn.execute(
            """
            SELECT * FROM jobs
            WHERE cluster_name = ? AND sky_job_id = ?
            """,
            (cluster_name, sky_job_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self.db._row_to_job(row)

    def _parse_sky_cluster(self, sky_cluster: dict) -> Cluster:
        """Parse SkyPilot cluster dict into Cluster model.

        Args:
            sky_cluster: Cluster dict from sky.status()

        Returns:
            Cluster object
        """
        # SkyPilot cluster format (approximate):
        # {
        #     'name': 'my-cluster',
        #     'status': 'UP',
        #     'resources': {...},
        #     'launched_at': ...,
        # }
        return Cluster(
            name=sky_cluster.get("name", "unknown"),
            status=sky_cluster.get("status", "UNKNOWN"),
            num_nodes=sky_cluster.get("num_nodes", 1),
            instance_type=sky_cluster.get("instance_type"),
            cloud=sky_cluster.get("cloud"),
            created_at=self._parse_timestamp(sky_cluster.get("launched_at")),
            last_seen=datetime.utcnow(),
        )

    def _parse_sky_job_status(self, sky_status: str) -> JobStatus:
        """Parse SkyPilot job status string to JobStatus enum.

        Args:
            sky_status: Status string from SkyPilot

        Returns:
            JobStatus enum value
        """
        # SkyPilot status strings (approximate)
        status_map = {
            "PENDING": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "SUCCEEDED": JobStatus.SUCCEEDED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.CANCELLED,
            "UNKNOWN": JobStatus.UNKNOWN,
        }
        return status_map.get(sky_status.upper(), JobStatus.UNKNOWN)

    def _parse_timestamp(self, timestamp: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime.

        Args:
            timestamp: Timestamp string

        Returns:
            datetime object or None
        """
        if not timestamp:
            return None
        try:
            # Try common formats
            return datetime.fromisoformat(timestamp)
        except (ValueError, AttributeError):
            return None
