"""Script to backfill policy versions for all checkpoints in the database."""

import asyncio
import logging
from datetime import datetime

from .database import Database
from .services import ObservatoryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill_checkpoint_versions(db_path: str | None = None):
    """Backfill policy versions for all checkpoints missing them.

    Args:
        db_path: Path to the SQLite database (default: ~/.skydeck/skydeck.db)
    """
    from pathlib import Path

    if db_path is None:
        db_path = str(Path.home() / ".skydeck" / "skydeck.db")
    db = Database(db_path)
    await db.connect()

    try:
        # Get all experiments
        experiments = await db.get_all_experiments()
        logger.info(f"Found {len(experiments)} experiments")

        total_checkpoints = 0
        updated_checkpoints = 0

        for exp in experiments:
            logger.info(f"Processing experiment: {exp.id}")

            # Get all checkpoints for this experiment (no limit)
            cursor = await db._conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE experiment_id = ?
                ORDER BY epoch DESC
                """,
                (exp.id,),
            )
            rows = await cursor.fetchall()
            checkpoints = [db._row_to_checkpoint(row) for row in rows]

            total_checkpoints += len(checkpoints)
            logger.info(f"  Found {len(checkpoints)} checkpoints")

            # Filter checkpoints that need backfill
            needs_backfill = [cp for cp in checkpoints if not cp.policy_version]

            if not needs_backfill:
                logger.info("  All checkpoints have versions, skipping")
                continue

            logger.info(f"  {len(needs_backfill)} checkpoints need version backfill")

            # Fetch policy version once per experiment
            policy_version = ObservatoryService.fetch_policy_version(exp.id)
            observatory_url = ObservatoryService.get_policy_api_url(exp.id, limit=500)

            if policy_version:
                logger.info(f"  Found policy version: {policy_version}")
            else:
                logger.warning(f"  Could not fetch policy version for {exp.id}")

            # Update all checkpoints that need backfill
            for cp in needs_backfill:
                # Update the checkpoint fields
                cp.policy_version = policy_version
                cp.observatory_url = observatory_url
                cp.synced_at = datetime.utcnow()

                # Save the updated checkpoint
                await db.save_checkpoint(cp)
                updated_checkpoints += 1

            logger.info(f"  Updated {len(needs_backfill)} checkpoints")

        logger.info("\nBackfill complete:")
        logger.info(f"  Total checkpoints: {total_checkpoints}")
        logger.info(f"  Updated checkpoints: {updated_checkpoints}")

    finally:
        await db.close()


async def main():
    """Main entry point."""
    await backfill_checkpoint_versions()


if __name__ == "__main__":
    asyncio.run(main())
