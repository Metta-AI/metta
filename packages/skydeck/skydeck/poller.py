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
        """Get managed jobs from SkyPilot API.

        Returns:
            List of job dicts from sky.jobs.queue()
        """
        try:
            import re

            # Use SkyPilot Python API to get jobs
            loop = asyncio.get_event_loop()

            def fetch_jobs():
                import sky
                request_id = sky.jobs.queue(refresh=False, skip_finished=False)
                return sky.get(request_id)

            sky_jobs = await loop.run_in_executor(None, fetch_jobs)
            logger.debug(f"Got {len(sky_jobs)} jobs from SkyPilot API")

            job_dicts = []
            for sky_job in sky_jobs:
                try:
                    job_id = str(sky_job.get("job_id") or sky_job.get("_job_id", ""))
                    if not job_id:
                        continue

                    job_name = sky_job.get("job_name") or sky_job.get("task_name", "unknown")

                    # Parse status from ManagedJobStatus enum string
                    status_str = str(sky_job.get("status", "UNKNOWN"))
                    # Extract status from "ManagedJobStatus.RUNNING" format
                    if "." in status_str:
                        status_str = status_str.split(".")[-1]
                    status = status_str.upper()

                    # Parse resources
                    nodes = 1
                    gpus = 0
                    resources_str = sky_job.get("resources", "")

                    # Parse "4x[A10G:4]" format
                    resource_match = re.search(r"(\d+)?x?\[([^\]]+)\]", resources_str)
                    if resource_match:
                        if resource_match.group(1):
                            nodes = int(resource_match.group(1))
                        # Parse GPU count from "A10G:4"
                        gpu_match = re.search(r"[A-Z0-9]+:(\d+)", resource_match.group(2))
                        if gpu_match:
                            gpus = int(gpu_match.group(1))

                    # Parse timestamps (they come as float strings)
                    submitted_at = None
                    start_at = None
                    end_at = None
                    try:
                        if sky_job.get("submitted_at"):
                            submitted_at = float(sky_job["submitted_at"])
                        if sky_job.get("start_at"):
                            start_at = float(sky_job["start_at"])
                        if sky_job.get("end_at"):
                            end_at = float(sky_job["end_at"])
                    except (ValueError, TypeError):
                        pass

                    job_dict = {
                        "id": job_id,
                        "name": job_name,
                        "status": status,
                        "submitted_at": submitted_at,
                        "start_at": start_at,
                        "end_at": end_at,
                        "job_duration": sky_job.get("job_duration", 0),
                        "resources": resources_str,
                        "cloud": sky_job.get("cloud", ""),
                        "infra": sky_job.get("infra", "managed"),
                        "entrypoint": sky_job.get("entrypoint", ""),
                        "nodes": nodes,
                        "gpus": gpus,
                    }
                    job_dicts.append(job_dict)

                except Exception as e:
                    logger.warning(f"Failed to parse job: {e}")
                    continue

            logger.info(f"Retrieved {len(job_dicts)} managed jobs from SkyPilot API")
            return job_dicts

        except Exception as e:
            logger.error(f"Error getting managed jobs from SkyPilot API: {e}", exc_info=True)
            return []
