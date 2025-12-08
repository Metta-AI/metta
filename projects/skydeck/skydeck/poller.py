"""Background poller for SkyPilot state."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .state_manager import StateManager

logger = logging.getLogger(__name__)


class Poller:
    """Background task that polls SkyPilot for cluster and job status.

    Runs periodically to update the cached state from actual SkyPilot infrastructure.
    """

    def __init__(self, state_manager: StateManager, interval: int = 30):
        """Initialize poller.

        Args:
            state_manager: StateManager instance to update
            interval: Polling interval in seconds (default: 30)
        """
        self.state_manager = state_manager
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.last_poll: Optional[datetime] = None

    async def start(self):
        """Start the background polling task."""
        if self.running:
            logger.warning("Poller already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._poll_loop())
        logger.info(f"Poller started (interval={self.interval}s)")

    async def stop(self):
        """Stop the background polling task."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Poller stopped")

    async def poll_once(self):
        """Execute one polling cycle immediately."""
        try:
            await self._poll()
            self.last_poll = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error during poll: {e}", exc_info=True)

    async def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                await self._poll()
                self.last_poll = datetime.utcnow()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during poll: {e}", exc_info=True)

            # Wait for next interval
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def _poll(self):
        """Execute one poll of SkyPilot state.

        This fetches managed jobs from SkyPilot.
        """
        logger.debug("Polling SkyPilot managed jobs...")

        # Poll managed jobs
        jobs = await self._get_managed_jobs()
        if jobs:
            await self.state_manager.update_managed_jobs(jobs)
            logger.debug(f"Updated {len(jobs)} managed jobs")

    async def _get_managed_jobs(self) -> list[dict]:
        """Get managed jobs from SkyPilot using local state database.

        Returns:
            List of job dicts from sky.jobs.state.get_managed_jobs()
        """
        try:
            import os

            import sky.jobs.state

            # Call get_managed_jobs() to directly read from local database
            loop = asyncio.get_event_loop()
            jobs = await loop.run_in_executor(None, lambda: sky.jobs.state.get_managed_jobs(job_id=None))

            if not jobs:
                logger.debug("No managed jobs found")
                return []

            # Get current username for filtering
            username = os.getenv("USER") or os.getenv("USERNAME")

            # Convert job dicts to simplified format for our database
            job_dicts = []
            for job in jobs:
                job_name = job.get("job_name", "unknown")

                # Filter to only jobs that belong to the current user
                # Jobs typically start with username prefix (e.g., "daveey.experiment")
                if username and not job_name.startswith(f"{username}."):
                    continue

                # Extract relevant fields from the managed job dict
                # The job dict already has the right structure from sky.jobs.state
                job_dict = {
                    "id": str(job.get("job_id")),
                    "name": job_name,
                    "status": str(job.get("status", "UNKNOWN")).replace("ManagedJobStatus.", ""),
                    "submitted_at": job.get("submitted_at"),
                    "start_at": job.get("start_at"),
                    "end_at": job.get("end_at"),
                    "job_duration": job.get("job_duration", 0),
                    "resources": job.get("resources", ""),
                    "cloud": job.get("cloud", ""),
                    "region": job.get("region", ""),
                    "zone": job.get("zone", ""),
                    "infra": job.get("infra", ""),
                    "accelerators": job.get("accelerators", {}),
                    "entrypoint": job.get("entrypoint", ""),
                }
                job_dicts.append(job_dict)

            logger.debug(f"Retrieved {len(job_dicts)} managed jobs from local database (filtered by user {username})")
            return job_dicts

        except ImportError:
            logger.warning("SkyPilot not installed, cannot poll jobs")
            return []
        except Exception as e:
            logger.error(f"Error getting managed jobs from SkyPilot: {e}", exc_info=True)
            return []
