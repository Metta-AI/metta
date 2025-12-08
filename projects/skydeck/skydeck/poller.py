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
        """Get managed jobs from SkyPilot using CLI.

        Returns:
            List of job dicts parsed from sky status output
        """
        try:
            import re
            import subprocess
            from datetime import datetime, timedelta

            # Run sky status to get managed jobs
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(["sky", "status"], capture_output=True, text=True, timeout=30)
            )

            if result.returncode != 0:
                logger.error(f"sky status failed with code {result.returncode}: {result.stderr}")
                return []

            output = result.stdout

            # Parse the output to extract managed jobs
            # Look for the "Managed jobs" section
            if "Managed jobs" not in output:
                logger.debug("No managed jobs section in sky status output")
                return []

            # Split output by sections
            lines = output.split("\n")

            # Find the managed jobs table
            in_managed_jobs = False
            job_dicts = []

            for _i, line in enumerate(lines):
                if "Managed jobs" in line:
                    in_managed_jobs = True
                    continue

                if in_managed_jobs:
                    # Check if we've reached the next section
                    if line.startswith("Services") or line.startswith("Clusters"):
                        break

                    # Skip header and separator lines
                    if "ID" in line and "TASK" in line:
                        continue
                    if not line.strip():
                        continue

                    # Parse job line (ID TASK NAME REQUESTED SUBMITTED DURATION ... STATUS POOL)
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    # Try to parse the job line
                    try:
                        job_id = parts[0]
                        # Skip if not a valid job ID (numeric)
                        if not job_id.isdigit():
                            continue

                        # Find the job name (look for pattern like daveey.xxx)
                        name_match = re.search(r"[a-zA-Z_][a-zA-Z0-9_.-]+", line)
                        job_name = name_match.group(0) if name_match else "unknown"

                        # Find status (typically near the end)
                        status = "UNKNOWN"
                        for word in parts:
                            if word.upper() in ["RUNNING", "PENDING", "FAILED", "SUCCEEDED", "CANCELLED"]:
                                status = word.upper()
                                break

                        # Parse resources (REQUESTED column)
                        # Format examples: "4x[A10G:4, L4:4]" or "1xCPU:4" or "[A10G:4]"
                        nodes = 1
                        gpus = 0
                        resources_str = ""

                        # Look for pattern like "4x[A10G:4]" or just "[A10G:4]"
                        resource_match = re.search(r"(\d+)?x?\[([^\]]+)\]", line)
                        if resource_match:
                            if resource_match.group(1):
                                nodes = int(resource_match.group(1))
                            resources_str = resource_match.group(2)

                            # Parse GPU count from patterns like "A10G:4" or "L4:4"
                            gpu_match = re.search(r"[A-Z0-9]+:(\d+)", resources_str)
                            if gpu_match:
                                gpus = int(gpu_match.group(1))

                        # Parse submitted time
                        submitted_at = None
                        time_match = re.search(r"(\d+)\s+(hrs?|mins?|days?|secs?)\s+ago", line)
                        if time_match:
                            value = int(time_match.group(1))
                            unit = time_match.group(2)
                            if "hr" in unit:
                                submitted_at = (datetime.utcnow() - timedelta(hours=value)).timestamp()
                            elif "min" in unit:
                                submitted_at = (datetime.utcnow() - timedelta(minutes=value)).timestamp()
                            elif "day" in unit:
                                submitted_at = (datetime.utcnow() - timedelta(days=value)).timestamp()
                            elif "sec" in unit:
                                submitted_at = (datetime.utcnow() - timedelta(seconds=value)).timestamp()

                        job_dict = {
                            "id": job_id,
                            "name": job_name,
                            "status": status,
                            "submitted_at": submitted_at,
                            "start_at": submitted_at,  # Approximation
                            "end_at": None if status in ["RUNNING", "PENDING"] else submitted_at,
                            "job_duration": 0,
                            "resources": resources_str,
                            "cloud": "",
                            "region": "",
                            "zone": "",
                            "infra": "managed",
                            "accelerators": {},
                            "entrypoint": "",
                            "nodes": nodes,
                            "gpus": gpus,
                        }
                        job_dicts.append(job_dict)
                        logger.debug(f"Parsed managed job: {job_id} - {job_name} ({status})")

                    except Exception as e:
                        logger.warning(f"Failed to parse job line: {line}: {e}")
                        continue

            logger.debug(f"Retrieved {len(job_dicts)} managed jobs from sky status CLI")
            return job_dicts

        except subprocess.TimeoutExpired:
            logger.error("sky status command timed out")
            return []
        except FileNotFoundError:
            logger.warning("sky command not found, cannot poll jobs")
            return []
        except Exception as e:
            logger.error(f"Error getting managed jobs from SkyPilot CLI: {e}", exc_info=True)
            return []
