"""Background syncer for experiment checkpoints."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from .database import Database
from .models import Checkpoint

logger = logging.getLogger(__name__)


class Syncer:
    """Background task that syncs checkpoints from S3.

    Loads checkpoints from S3 for all experiments with run_name set.
    """

    def __init__(self, db: Database, interval: int = 60):
        """Initialize syncer.

        Args:
            db: Database instance
            interval: Sync interval in seconds (default: 60)
        """
        self.db = db
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.last_sync: Optional[datetime] = None

    async def start(self):
        """Start the background syncing task."""
        if self.running:
            logger.warning("Syncer already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._sync_loop())
        logger.info(f"Syncer started (interval={self.interval}s)")

    async def stop(self):
        """Stop the background syncing task."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Syncer stopped")

    async def sync_once(self):
        """Execute one sync cycle immediately."""
        try:
            await self._sync()
            self.last_sync = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error during sync: {e}", exc_info=True)

    async def _sync_loop(self):
        """Main syncing loop."""
        while self.running:
            try:
                await self._sync()
                self.last_sync = datetime.utcnow()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during sync: {e}", exc_info=True)

            # Wait for next interval
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def _sync(self):
        """Execute one sync of experiment checkpoints from S3."""
        logger.debug("Syncing experiment checkpoints from S3...")

        # Get all experiments
        experiments = await self.db.get_all_experiments()

        if not experiments:
            logger.debug("No experiments to sync")
            return

        # Sync checkpoints for all experiments
        synced_count = 0
        for exp in experiments:
            try:
                await self._sync_experiment(exp.id)
                synced_count += 1
            except Exception as e:
                logger.error(f"Error syncing experiment {exp.id}: {e}", exc_info=True)

        logger.debug(f"Synced {synced_count} experiments from S3")

    async def _sync_experiment(self, experiment_id: str):
        """Sync checkpoints for a single experiment from S3.

        Lists .mpt files from S3: softmax-public/policies/{experiment_id}/
        """
        # Get experiment
        exp = await self.db.get_experiment(experiment_id)
        if not exp:
            logger.debug(f"Experiment {experiment_id} not found, skipping sync")
            return

        s3_path = f"s3://softmax-public/policies/{experiment_id}/"

        try:
            # Run aws s3 ls to list checkpoints
            process = await asyncio.create_subprocess_exec(
                "aws",
                "s3",
                "ls",
                s3_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                if stderr:
                    error_msg = stderr.decode().strip()
                    # Don't log error if path doesn't exist (no checkpoints yet)
                    if "does not exist" not in error_msg.lower():
                        logger.warning(f"Error listing S3 path {s3_path}: {error_msg}")
                return

            # Parse output to find .mpt files
            checkpoints = self._parse_s3_output(stdout.decode(), experiment_id, s3_path)

            # Save checkpoints to database
            for checkpoint in checkpoints:
                await self.db.save_checkpoint(checkpoint)

            if checkpoints:
                logger.debug(f"Found {len(checkpoints)} checkpoints for {experiment_id} in {s3_path}")

        except Exception as e:
            logger.error(f"Error syncing experiment {experiment_id} from S3: {e}", exc_info=True)

    def _parse_s3_output(self, output: str, experiment_id: str, s3_path: str) -> list[Checkpoint]:
        """Parse aws s3 ls output to extract .mpt checkpoint files.

        Args:
            output: stdout from aws s3 ls command
            experiment_id: Experiment ID
            s3_path: S3 path that was listed

        Returns:
            List of Checkpoint objects

        Example output line:
            2023-01-15 10:30:45     123456 checkpoint_1000.mpt
        """
        checkpoints = []

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            # Parse line format: DATE TIME SIZE FILENAME
            parts = line.split()
            if len(parts) < 4:
                continue

            filename = parts[3]
            if not filename.endswith(".mpt"):
                continue

            # Extract epoch/step number from filename
            # Try patterns: v100.mpt, checkpoint_1000.mpt, epoch_5.mpt, model_1000.mpt, 1000.mpt
            epoch = None
            patterns = [
                r":?v(\d+)\.mpt",  # v100.mpt or name:v100.mpt
                r"checkpoint[_-]?(\d+)",
                r"epoch[_-]?(\d+)",
                r"model[_-]?(\d+)",
                r"step[_-]?(\d+)",
                r"^(\d+)\.mpt$",  # Just a number
            ]

            for pattern in patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    epoch = int(match.group(1))
                    break

            if epoch is None:
                # If we can't parse epoch, use filename hash as epoch
                logger.debug(f"Could not parse epoch from filename: {filename}, skipping")
                continue

            # Parse timestamp
            try:
                date_str = f"{parts[0]} {parts[1]}"
                created_at = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = datetime.utcnow()

            # Build full S3 path
            model_path = f"{s3_path}{filename}"

            checkpoint = Checkpoint(
                experiment_id=experiment_id,
                epoch=epoch,
                model_path=model_path,
                replay_paths=[],  # No replay files in S3 listing
                metrics={},
                created_at=created_at,
                synced_at=datetime.utcnow(),
            )
            checkpoints.append(checkpoint)

        return checkpoints
