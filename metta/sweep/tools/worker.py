"""Worker tool for distributed job execution via PostgreSQL queue."""

import json
import logging
import os
import socket
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.sweep.dispatchers import LocalDispatcher
from metta.sweep.models import JobDefinition

logger = logging.getLogger(__name__)


class WorkerTool(Tool):
    """Worker node that polls PostgreSQL queue and executes jobs locally.

    This tool runs on remote workers to:
    1. Poll a PostgreSQL job queue for pending jobs
    2. Claim jobs atomically using row-level locking
    3. Execute jobs using LocalDispatcher
    4. Update job status in the database
    """

    # Database configuration (can be passed as argument or environment variable)
    db_url: Optional[str] = None  # PostgreSQL connection URL
    run: Optional[str] = None

    # Worker configuration
    worker_id: Optional[str] = None  # Unique worker identifier
    group: Optional[str] = None  # Group/experiment to process jobs for (None = any group)
    poll_interval: int = 10  # Seconds between polls when queue is empty
    heartbeat_interval: int = 30  # Seconds between worker heartbeats
    idle_timeout: int = 600  # Seconds of idle time before worker shuts down (default: 10 minutes, 0 = never)
    max_retries: int = 3  # Maximum retries for failed jobs

    # Local execution settings
    capture_output: bool = True  # Whether to capture subprocess output

    def invoke(self, args: dict[str, str]) -> int | None:
        """Main entry point for the worker tool."""
        init_logging()

        # Detect multi-node environment (use local variables, not self attributes)
        node_rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
        num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
        num_gpus = int(os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", "1"))
        is_master = node_rank == 0
        is_distributed = num_nodes > 1

        logger.info(f"Node configuration: rank={node_rank}/{num_nodes} nodes, {num_gpus} GPUs, master={is_master}")

        # Set defaults from environment if not provided
        if not self.db_url:
            self.db_url = os.environ.get("POSTGRES_URL")
            if not self.db_url:
                raise ValueError("Database URL required: provide db_url or set POSTGRES_URL environment variable")

        if not self.worker_id:
            # Include node rank in worker ID for multi-node setups
            base_id = os.environ.get("WORKER_ID", socket.gethostname())
            if is_distributed:
                self.worker_id = f"{base_id}-node{node_rank}"
            else:
                self.worker_id = base_id

        # Initialize local dispatcher for job execution (store as local variable)
        # Use torchrun (via run.sh) for multi-GPU or multi-node execution
        use_torchrun = is_distributed or num_gpus > 1
        local_dispatcher = LocalDispatcher(capture_output=self.capture_output, torchrun=use_torchrun)

        # Track last heartbeat and last job time for idle timeout
        last_heartbeat = time.time()
        last_job_time = time.time()  # Track when we last processed a job
        poll_count = 0  # Track number of polls for status message

        logger.info(f"Worker {self.worker_id} starting, polling {self.db_url}")
        if self.group:
            logger.info(f"Worker {self.worker_id} assigned to group: {self.group}")
        else:
            logger.info(f"Worker {self.worker_id} will process jobs from any group")
        logger.info(f"Worker {self.worker_id} initialized with poll_interval={self.poll_interval}s")

        if self.idle_timeout > 0:
            logger.info(f"Worker {self.worker_id} will shut down after {self.idle_timeout}s of idle time")
        else:
            logger.info(f"Worker {self.worker_id} will run indefinitely (idle_timeout=0)")

        # Register worker in database (only master in distributed mode)
        if not is_distributed or is_master:
            self._register_worker()

        try:
            # Different behavior for master vs worker nodes in distributed mode
            if is_distributed and not is_master:
                # Worker nodes: wait for job signals from master
                logger.info(f"Worker node {node_rank} waiting for job signals from master...")
                self._run_worker_node(node_rank, num_nodes, num_gpus, local_dispatcher)
            else:
                # Master node (or single-node): normal database polling
                logger.info(f"{'Master' if is_distributed else 'Worker'} polling database for jobs...")
                # Main polling loop
                while True:
                    try:
                        # Check idle timeout
                        if self.idle_timeout > 0:
                            idle_duration = time.time() - last_job_time
                            if idle_duration >= self.idle_timeout:
                                logger.info(
                                    f"Worker {self.worker_id} idle for {int(idle_duration)}s "
                                    f"(>= {self.idle_timeout}s timeout), shutting down..."
                                )
                                break

                        # Send heartbeat if needed
                        if time.time() - last_heartbeat > self.heartbeat_interval:
                            self._send_heartbeat()
                            last_heartbeat = time.time()

                        # Try to claim and execute a job
                        job = self._claim_job()
                        if job:
                            logger.info(f"Worker {self.worker_id} claimed job {job.run_id}")

                            # Signal other nodes if in distributed mode
                            if is_distributed:
                                self._signal_job_to_workers(job, node_rank, num_nodes)

                            # Execute the job
                            self._execute_job(job, local_dispatcher, is_distributed, num_nodes, num_gpus)
                            # Update heartbeat and last job time after execution
                            last_heartbeat = time.time()
                            last_job_time = time.time()
                            poll_count = 0  # Reset poll count after job
                        else:
                            # No jobs available - show polling status
                            poll_count += 1
                            idle_time = int(time.time() - last_job_time)
                            group_msg = f"[group={self.group}]" if self.group else "[any group]"

                            # Show status every 3 polls (30 seconds by default)
                            if poll_count % 3 == 1:
                                remaining = ""
                                if self.idle_timeout > 0:
                                    remaining_time = max(0, self.idle_timeout - idle_time)
                                    remaining = f", shutting down in {remaining_time}s"

                                logger.info(
                                    f"Worker {self.worker_id} {group_msg}: Polling... "
                                    f"(idle for {idle_time}s{remaining})"
                                )

                            # Wait before polling again
                            time.sleep(self.poll_interval)

                    except KeyboardInterrupt:
                        logger.info(f"Worker {self.worker_id} received interrupt, shutting down...")
                        break
                    except Exception as e:
                        logger.error(f"Error in worker loop: {e}", exc_info=True)
                        time.sleep(self.poll_interval)

        finally:
            # Mark worker as offline
            self._unregister_worker()

        return 0

    def _register_worker(self):
        """Register this worker in the database."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Create worker_status table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS worker_status (
                            worker_id VARCHAR(255) PRIMARY KEY,
                            hostname VARCHAR(255),
                            group_id VARCHAR(255),
                            last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            status VARCHAR(50) DEFAULT 'idle',
                            current_job_id VARCHAR(255),
                            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            metadata JSONB
                        );
                    """)

                    # Add group_id column if it doesn't exist (for backward compatibility)
                    cursor.execute("""
                        ALTER TABLE worker_status
                        ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
                    """)

                    # Register or update this worker
                    cursor.execute(
                        """
                        INSERT INTO worker_status (worker_id, hostname, group_id, status)
                        VALUES (%s, %s, %s, 'idle')
                        ON CONFLICT (worker_id) DO UPDATE
                        SET hostname = EXCLUDED.hostname,
                            group_id = EXCLUDED.group_id,
                            status = 'idle',
                            last_heartbeat = NOW(),
                            started_at = NOW(),
                            current_job_id = NULL
                    """,
                        (self.worker_id, socket.gethostname(), self.group),
                    )
                    conn.commit()

            logger.info(f"Worker {self.worker_id} registered in database")
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")

    def _unregister_worker(self):
        """Mark this worker as offline in the database."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE worker_status
                        SET status = 'offline',
                            last_heartbeat = NOW(),
                            current_job_id = NULL
                        WHERE worker_id = %s
                    """,
                        (self.worker_id,),
                    )
                    conn.commit()

            logger.info(f"Worker {self.worker_id} marked as offline")
        except Exception as e:
            logger.error(f"Failed to unregister worker: {e}")

    def _send_heartbeat(self):
        """Send a heartbeat to the database to indicate worker is alive."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE worker_status
                        SET last_heartbeat = NOW()
                        WHERE worker_id = %s
                    """,
                        (self.worker_id,),
                    )
                    conn.commit()

            logger.debug(f"Worker {self.worker_id} sent heartbeat")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    def _claim_job(self) -> Optional[JobDefinition]:
        """Atomically claim a pending job from the queue.

        Uses PostgreSQL's FOR UPDATE SKIP LOCKED to ensure only one worker
        can claim each job. If worker has a group, only claims jobs from that group.

        Returns:
            JobDefinition if a job was claimed, None if no jobs available.
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Add group_id column if it doesn't exist (for backward compatibility)
                    cursor.execute("""
                        ALTER TABLE job_queue
                        ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
                    """)

                    # Atomically claim the oldest pending job (optionally filtered by group)
                    if self.group:
                        # Only claim jobs from the specified group
                        cursor.execute(
                            """
                            UPDATE job_queue
                            SET status = 'claimed',
                                worker_id = %s,
                                claimed_at = NOW()
                            WHERE id = (
                                SELECT id FROM job_queue
                                WHERE status = 'pending'
                                AND group_id = %s
                                ORDER BY created_at
                                FOR UPDATE SKIP LOCKED
                                LIMIT 1
                            )
                            RETURNING job_id, job_definition
                        """,
                            (self.worker_id, self.group),
                        )
                    else:
                        # Claim any pending job (no group filter)
                        cursor.execute(
                            """
                            UPDATE job_queue
                            SET status = 'claimed',
                                worker_id = %s,
                                claimed_at = NOW()
                            WHERE id = (
                                SELECT id FROM job_queue
                                WHERE status = 'pending'
                                ORDER BY created_at
                                FOR UPDATE SKIP LOCKED
                                LIMIT 1
                            )
                            RETURNING job_id, job_definition
                        """,
                            (self.worker_id,),
                        )

                    row = cursor.fetchone()
                    if row:
                        # Update worker status to busy
                        cursor.execute(
                            """
                            UPDATE worker_status
                            SET status = 'busy',
                                current_job_id = %s,
                                last_heartbeat = NOW()
                            WHERE worker_id = %s
                        """,
                            (row["job_id"], self.worker_id),
                        )
                        conn.commit()

                        # Deserialize job definition
                        job_dict = row["job_definition"]
                        # Convert ISO format string back to datetime if present
                        if "created_at" in job_dict and isinstance(job_dict["created_at"], str):
                            job_dict["created_at"] = datetime.fromisoformat(job_dict["created_at"])

                        return JobDefinition(**job_dict)

            return None

        except Exception as e:
            logger.error(f"Failed to claim job: {e}", exc_info=True)
            return None

    def _execute_job(
        self, job: JobDefinition, local_dispatcher: LocalDispatcher, is_distributed: bool, num_nodes: int, num_gpus: int
    ):
        """Execute a job using appropriate strategy (run.sh for distributed, direct for single).

        Args:
            job: The job definition to execute.
            local_dispatcher: The LocalDispatcher instance to use for execution.
            is_distributed: Whether we're in multi-node mode.
            num_nodes: Total number of nodes.
            num_gpus: Number of GPUs per node.
        """
        try:
            # Update status to running
            self._update_job_status(job.run_id, "running", is_distributed, is_master=True)
            self._update_worker_status("busy", job.run_id)

            # LocalDispatcher handles both torchrun and direct execution based on its configuration
            if is_distributed or num_gpus > 1:
                logger.info(f"Executing distributed job {job.run_id}: {job.cmd} (nodes={num_nodes}, gpus={num_gpus})")
            else:
                logger.info(f"Executing job {job.run_id}: {job.cmd}")

            dispatch_id = local_dispatcher.dispatch(job)
            logger.info(f"Job {job.run_id} dispatched with PID {dispatch_id}")

            # Monitor the subprocess until completion
            while True:
                active = local_dispatcher.check_processes()
                if active == 0:
                    break
                time.sleep(5)  # Check every 5 seconds

                # Send heartbeat while job is running
                self._send_heartbeat()

            # Job completed successfully
            self._update_job_status(job.run_id, "completed", is_distributed, is_master=True)
            logger.info(f"Job {job.run_id} completed successfully")

        except Exception as e:
            error_msg = f"Job {job.run_id} failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._update_job_status(job.run_id, "failed", is_distributed, is_master=True, error_message=error_msg)

        finally:
            # Update worker status back to idle
            self._update_worker_status("idle")

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        is_distributed: bool = False,
        is_master: bool = True,
        error_message: Optional[str] = None,
    ):
        """Update the status of a job in the database.

        Args:
            job_id: The job ID to update.
            status: The new status ('running', 'completed', 'failed').
            is_distributed: Whether we're in distributed mode.
            is_master: Whether this is the master node.
            error_message: Optional error message if job failed.
        """
        # Only master updates job status in distributed mode
        if is_distributed and not is_master:
            logger.debug("Worker node skipping job status update (master's responsibility)")
            return

        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    if status == "running":
                        cursor.execute(
                            """
                            UPDATE job_queue
                            SET status = %s,
                                started_at = NOW()
                            WHERE job_id = %s
                        """,
                            (status, job_id),
                        )
                    elif status in ("completed", "failed"):
                        cursor.execute(
                            """
                            UPDATE job_queue
                            SET status = %s,
                                completed_at = NOW(),
                                error_message = %s
                            WHERE job_id = %s
                        """,
                            (status, error_message, job_id),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE job_queue
                            SET status = %s
                            WHERE job_id = %s
                        """,
                            (status, job_id),
                        )
                    conn.commit()

            logger.debug(f"Updated job {job_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    def _update_worker_status(self, status: str, current_job_id: Optional[str] = None):
        """Update worker status in the database.

        Args:
            status: The new worker status ('idle', 'busy', 'offline').
            current_job_id: Optional current job ID if worker is busy.
        """
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE worker_status
                        SET status = %s,
                            current_job_id = %s,
                            last_heartbeat = NOW()
                        WHERE worker_id = %s
                    """,
                        (status, current_job_id, self.worker_id),
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to update worker status: {e}")

    def _run_worker_node(self, node_rank: int, num_nodes: int, num_gpus: int, local_dispatcher: LocalDispatcher):
        """Worker node loop: wait for job signals from master.

        Args:
            node_rank: This worker's rank in the cluster.
            num_nodes: Total number of nodes.
            num_gpus: Number of GPUs per node.
            local_dispatcher: The LocalDispatcher instance for executing jobs.
        """
        last_job_time = time.time()
        processed_jobs = set()  # Track processed job IDs to avoid re-execution

        while True:
            try:
                # Check idle timeout
                if self.idle_timeout > 0:
                    idle_duration = time.time() - last_job_time
                    if idle_duration >= self.idle_timeout:
                        logger.info(
                            f"Worker node {node_rank} idle for {int(idle_duration)}s "
                            f"(>= {self.idle_timeout}s timeout), shutting down..."
                        )
                        break

                # Check for job signals in database
                try:
                    with psycopg2.connect(self.db_url) as conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                            # Get unexpired job signals
                            cursor.execute("""
                                SELECT job_id, job_definition
                                FROM job_signals
                                WHERE expires_at > NOW()
                                ORDER BY created_at DESC
                                LIMIT 1
                            """)
                            row = cursor.fetchone()

                            if row and row["job_id"] not in processed_jobs:
                                job_dict = row["job_definition"]
                                # Convert ISO format string back to datetime if present
                                if "created_at" in job_dict and isinstance(job_dict["created_at"], str):
                                    job_dict["created_at"] = datetime.fromisoformat(job_dict["created_at"])

                                job = JobDefinition(**job_dict)
                                logger.info(f"Worker node {node_rank} received job signal for {job.run_id}")

                                # Mark as processed immediately to avoid re-execution
                                processed_jobs.add(job.run_id)

                                # Execute the job using LocalDispatcher (which will use run.sh)
                                dispatch_id = local_dispatcher.dispatch(job)
                                logger.info(
                                    f"Worker node {node_rank} dispatched job {job.run_id} with PID {dispatch_id}"
                                )

                                # Wait for completion
                                while local_dispatcher.check_processes() > 0:
                                    time.sleep(5)

                                logger.info(f"Worker node {node_rank} completed job {job.run_id}")
                                last_job_time = time.time()

                                # Clean up the signal after execution
                                cursor.execute(
                                    """
                                    DELETE FROM job_signals WHERE job_id = %s
                                """,
                                    (job.run_id,),
                                )
                                conn.commit()

                except Exception as e:
                    logger.error(f"Error checking job signals: {e}")

                # Sleep briefly before next check
                time.sleep(2)  # Poll every 2 seconds

            except KeyboardInterrupt:
                logger.info(f"Worker node {node_rank} interrupted")
                break
            except Exception as e:
                logger.error(f"Worker node {node_rank} error: {e}", exc_info=True)
                time.sleep(5)

    def _signal_job_to_workers(self, job: JobDefinition, node_rank: int, num_nodes: int):
        """Signal other nodes to start the job via PostgreSQL database.

        Args:
            job: The job to signal.
            node_rank: Master node's rank (should be 0).
            num_nodes: Total number of nodes in cluster.
        """
        try:
            # Serialize job with datetime handling
            job_dict = asdict(job)
            if "created_at" in job_dict and isinstance(job_dict["created_at"], datetime):
                job_dict["created_at"] = job_dict["created_at"].isoformat()

            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Create table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS job_signals (
                            id SERIAL PRIMARY KEY,
                            job_id VARCHAR(255) NOT NULL,
                            job_definition JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT NOW(),
                            expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '5 minutes'),
                            CONSTRAINT unique_job_signal UNIQUE (job_id)
                        );
                    """)

                    # Insert or update signal
                    cursor.execute(
                        """
                        INSERT INTO job_signals (job_id, job_definition)
                        VALUES (%s, %s)
                        ON CONFLICT (job_id) DO UPDATE
                        SET job_definition = EXCLUDED.job_definition,
                            created_at = NOW(),
                            expires_at = NOW() + INTERVAL '5 minutes'
                    """,
                        (job.run_id, json.dumps(job_dict)),
                    )

                    # Clean up old signals
                    cursor.execute("DELETE FROM job_signals WHERE expires_at < NOW()")

                    conn.commit()

            logger.info(f"Master signaled job {job.run_id} to {num_nodes - 1} worker nodes via database")

            # Give workers a moment to poll
            time.sleep(2)

        except Exception as e:
            logger.error(f"Failed to signal job to workers: {e}", exc_info=True)
