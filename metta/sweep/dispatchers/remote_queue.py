"""Remote queue dispatcher for distributed job execution via PostgreSQL."""

import json
import logging
import os
from dataclasses import asdict
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from metta.adaptive.models import JobDefinition
from metta.adaptive.utils import get_display_id

logger = logging.getLogger(__name__)


class RemoteQueueDispatcher:
    """Dispatches jobs to a PostgreSQL queue for remote workers to execute."""

    def __init__(self, db_url: Optional[str] = None, group: Optional[str] = None):
        """Initialize the remote queue dispatcher.

        Args:
            db_url: PostgreSQL connection URL. If not provided, uses POSTGRES_URL env var.
            group: Optional group/experiment identifier to segregate jobs.
        """
        self.db_url = db_url or os.environ.get("POSTGRES_URL")
        if not self.db_url:
            raise ValueError("Database URL required: provide db_url or set POSTGRES_URL environment variable")

        self.group = group
        if self.group:
            logger.info(f"RemoteQueueDispatcher initialized for group: {self.group}")
        else:
            logger.info(f"RemoteQueueDispatcher initialized with database connection")
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure the job_queue table exists."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_name = 'job_queue'
                        );
                    """)
                    exists = cursor.fetchone()[0]
                    if not exists:
                        logger.warning("job_queue table does not exist. Please run metta/sweep/database/setup.py")
        except Exception as e:
            logger.error(f"Failed to check for job_queue table: {e}")
            raise

    def dispatch(self, job: JobDefinition) -> str:
        """Queue a job to the PostgreSQL database for remote execution.

        Args:
            job: The job definition to dispatch.

        Returns:
            The job ID (same as job.run_id for consistency).
        """
        display_id = get_display_id(job.run_id)

        try:
            # Serialize job definition to JSON
            # Convert datetime objects to ISO format strings for JSON serialization
            job_dict = asdict(job)
            if 'created_at' in job_dict and hasattr(job_dict['created_at'], 'isoformat'):
                job_dict['created_at'] = job_dict['created_at'].isoformat()

            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Add group_id column if it doesn't exist (for backward compatibility)
                    cursor.execute("""
                        ALTER TABLE job_queue
                        ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
                    """)

                    cursor.execute(
                        """
                        INSERT INTO job_queue (job_id, job_definition, group_id, status)
                        VALUES (%s, %s, %s, 'pending')
                        ON CONFLICT (job_id) DO UPDATE
                        SET job_definition = EXCLUDED.job_definition,
                            group_id = EXCLUDED.group_id,
                            status = 'pending',
                            created_at = NOW()
                        """,
                        (job.run_id, json.dumps(job_dict), self.group)
                    )
                    conn.commit()

            if self.group:
                logger.info(f"Queued job {display_id} to remote queue (group: {self.group})")
            else:
                logger.info(f"Queued job {display_id} to remote queue")
            return job.run_id

        except Exception as e:
            logger.error(f"Failed to queue job {job.run_id}: {e}", exc_info=True)
            raise

    def get_queue_stats(self) -> dict:
        """Get statistics about the current queue state.

        Returns:
            Dictionary with queue statistics.
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if self.group:
                        cursor.execute("""
                            SELECT
                                status,
                                COUNT(*) as count,
                                MIN(created_at) as oldest_job,
                                MAX(created_at) as newest_job
                            FROM job_queue
                            WHERE group_id = %s
                            GROUP BY status
                        """, (self.group,))
                    else:
                        cursor.execute("""
                            SELECT
                                status,
                                COUNT(*) as count,
                                MIN(created_at) as oldest_job,
                                MAX(created_at) as newest_job
                            FROM job_queue
                            GROUP BY status
                        """)
                    stats = cursor.fetchall()

                    # Also get worker status if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_name = 'worker_status'
                        );
                    """)
                    has_worker_table = cursor.fetchone()['exists']

                    worker_stats = []
                    if has_worker_table:
                        cursor.execute("""
                            SELECT
                                worker_id,
                                status,
                                last_heartbeat,
                                current_job_id
                            FROM worker_status
                            ORDER BY last_heartbeat DESC
                        """)
                        worker_stats = cursor.fetchall()

                    return {
                        'queue_stats': stats,
                        'worker_stats': worker_stats
                    }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'error': str(e)}

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled in the queue.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if job was successfully cancelled, False otherwise.
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE job_queue
                        SET status = 'failed',
                            error_message = 'Cancelled by controller',
                            completed_at = NOW()
                        WHERE job_id = %s
                        AND status IN ('pending', 'claimed')
                        """,
                        (job_id,)
                    )
                    cancelled = cursor.rowcount > 0
                    conn.commit()

            if cancelled:
                logger.info(f"Cancelled job {job_id}")
            else:
                logger.warning(f"Could not cancel job {job_id} - may already be running or completed")

            return cancelled

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False