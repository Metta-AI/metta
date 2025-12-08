"""Database layer for SkyDeck dashboard using SQLite."""

import json
from datetime import datetime
from typing import Optional

import aiosqlite

from .models import Checkpoint, Cluster, DesiredState, Experiment, Job, JobStatus


class Database:
    """Async SQLite database manager for SkyDeck."""

    def __init__(self, db_path: str = "skydeck.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Connect to database and create tables if needed."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._create_tables()
        await self._run_migrations()

    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        await self._conn.executescript(
            """
            -- Experiments table
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                desired_state TEXT NOT NULL,
                current_state TEXT NOT NULL,
                flags TEXT NOT NULL,  -- JSON
                base_command TEXT NOT NULL,
                run_name TEXT,
                git_branch TEXT,
                current_job_id TEXT,
                cluster_name TEXT,
                nodes INTEGER NOT NULL DEFAULT 1,
                gpus INTEGER NOT NULL DEFAULT 0,
                instance_type TEXT,
                cloud TEXT,
                region TEXT,
                zone TEXT,
                spot INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                wandb_link TEXT,
                description TEXT,
                tags TEXT NOT NULL,  -- JSON array
                exp_group TEXT,  -- Group name (renamed from 'group' to avoid SQL keyword)
                exp_order INTEGER NOT NULL DEFAULT 0  -- Display order within group
            );

            -- Jobs table (one-to-many with experiments)
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                cluster_name TEXT NOT NULL,
                sky_job_id INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                submitted_at TEXT,
                started_at TEXT,
                ended_at TEXT,
                command TEXT NOT NULL,
                logs_path TEXT,
                exit_code INTEGER,
                error_message TEXT,
                nodes INTEGER NOT NULL DEFAULT 1,
                gpus INTEGER NOT NULL DEFAULT 0,
                instance_type TEXT,
                cloud TEXT,
                region TEXT,
                zone TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            -- User settings table
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,  -- JSON
                updated_at TEXT NOT NULL
            );

            -- Clusters table (cached SkyPilot cluster info)
            CREATE TABLE IF NOT EXISTS clusters (
                name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                num_nodes INTEGER NOT NULL DEFAULT 0,
                instance_type TEXT,
                cloud TEXT,
                region TEXT,
                zone TEXT,
                created_at TEXT,
                last_seen TEXT NOT NULL
            );

            -- Flag schemas table (for typeahead)
            CREATE TABLE IF NOT EXISTS flag_schemas (
                flag_name TEXT PRIMARY KEY,
                flag_type TEXT NOT NULL,
                description TEXT,
                default_value TEXT,
                choices TEXT,  -- JSON array for enum types
                category TEXT
            );

            -- Checkpoints table (tracking model and replay files per epoch)
            CREATE TABLE IF NOT EXISTS checkpoints (
                experiment_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                model_path TEXT,
                replay_paths TEXT NOT NULL,  -- JSON array
                metrics TEXT NOT NULL,  -- JSON object
                created_at TEXT NOT NULL,
                synced_at TEXT NOT NULL,
                PRIMARY KEY (experiment_id, epoch),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_jobs_experiment_id ON jobs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_experiments_desired_state ON experiments(desired_state);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_id ON checkpoints(experiment_id);
            """
        )
        await self._conn.commit()

    async def _run_migrations(self):
        """Run database migrations for schema updates."""
        # Migration: Add git_branch column to experiments table if it doesn't exist
        cursor = await self._conn.execute("PRAGMA table_info(experiments)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]

        if "git_branch" not in column_names:
            await self._conn.execute("ALTER TABLE experiments ADD COLUMN git_branch TEXT")
            await self._conn.commit()

        # Migration: Add is_expanded column to experiments table if it doesn't exist
        if "is_expanded" not in column_names:
            await self._conn.execute("ALTER TABLE experiments ADD COLUMN is_expanded INTEGER NOT NULL DEFAULT 0")
            await self._conn.commit()

        # Migration: Add tool_path column to experiments table if it doesn't exist
        if "tool_path" not in column_names:
            await self._conn.execute("ALTER TABLE experiments ADD COLUMN tool_path TEXT")
            await self._conn.commit()

        # Migration: Add starred column to experiments table if it doesn't exist
        if "starred" not in column_names:
            await self._conn.execute("ALTER TABLE experiments ADD COLUMN starred INTEGER NOT NULL DEFAULT 0")
            await self._conn.commit()

    # Experiment operations

    async def save_experiment(self, experiment: Experiment):
        """Save or update an experiment."""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                id, name, desired_state, current_state, flags, base_command, tool_path, run_name, git_branch,
                current_job_id, cluster_name, nodes, gpus, instance_type, cloud,
                region, zone, spot, created_at, updated_at, wandb_link, description, tags,
                exp_group, exp_order, is_expanded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment.id,
                experiment.name,
                experiment.desired_state.value,
                experiment.current_state.value,
                json.dumps(experiment.flags),
                experiment.base_command,
                experiment.tool_path,
                experiment.run_name,
                experiment.git_branch,
                experiment.current_job_id,
                experiment.cluster_name,
                experiment.nodes,
                experiment.gpus,
                experiment.instance_type,
                experiment.cloud,
                experiment.region,
                experiment.zone,
                1 if experiment.spot else 0,
                experiment.created_at.isoformat(),
                experiment.updated_at.isoformat(),
                experiment.wandb_link,
                experiment.description,
                json.dumps(experiment.tags),
                experiment.group,
                experiment.order,
                1 if experiment.is_expanded else 0,
            ),
        )
        await self._conn.commit()

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        cursor = await self._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    async def get_all_experiments(self) -> list[Experiment]:
        """Get all experiments."""
        cursor = await self._conn.execute("SELECT * FROM experiments ORDER BY exp_order ASC, created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_experiment(row) for row in rows]

    async def delete_experiment(self, experiment_id: str):
        """Delete experiment and all its jobs (cascade)."""
        await self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        await self._conn.commit()

    async def update_experiment_state(
        self, experiment_id: str, current_state: JobStatus, current_job_id: Optional[str] = None
    ):
        """Update experiment current state and optionally current job."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET current_state = ?, current_job_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (current_state.value, current_job_id, datetime.utcnow().isoformat(), experiment_id),
        )
        await self._conn.commit()

    async def update_experiment_desired_state(self, experiment_id: str, desired_state: DesiredState):
        """Update experiment desired state."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET desired_state = ?, updated_at = ?
            WHERE id = ?
            """,
            (desired_state.value, datetime.utcnow().isoformat(), experiment_id),
        )
        await self._conn.commit()

    async def update_experiment_flags(self, experiment_id: str, flags: dict):
        """Update experiment flags."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET flags = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(flags), datetime.utcnow().isoformat(), experiment_id),
        )
        await self._conn.commit()

    async def set_experiment_cluster(self, experiment_id: str, cluster_name: str):
        """Set experiment cluster name."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET cluster_name = ?, updated_at = ?
            WHERE id = ?
            """,
            (cluster_name, datetime.utcnow().isoformat(), experiment_id),
        )
        await self._conn.commit()

    async def update_experiment_expanded(self, experiment_id: str, is_expanded: bool):
        """Update experiment expanded state."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET is_expanded = ?
            WHERE id = ?
            """,
            (1 if is_expanded else 0, experiment_id),
        )
        await self._conn.commit()

    async def rename_experiment(self, old_id: str, new_id: str):
        """Rename an experiment (change its ID).

        This updates the experiment ID and all references to it in jobs.
        """
        # Update the experiment ID
        await self._conn.execute(
            """
            UPDATE experiments
            SET id = ?
            WHERE id = ?
            """,
            (new_id, old_id),
        )

        # Update all jobs that reference this experiment
        await self._conn.execute(
            """
            UPDATE jobs
            SET experiment_id = ?
            WHERE experiment_id = ?
            """,
            (new_id, old_id),
        )

        await self._conn.commit()

    # Job operations

    async def save_job(self, job: Job, allow_update: bool = True):
        """Save or update a job.

        Args:
            job: Job to save
            allow_update: If False, will raise error if job ID already exists

        Raises:
            ValueError: If job ID already exists and allow_update is False
        """
        # Check if job already exists
        existing = await self.get_job(job.id)

        if existing and not allow_update:
            raise ValueError(f"Job with ID '{job.id}' already exists. Cannot create duplicate job.")

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                id, experiment_id, cluster_name, sky_job_id, status,
                created_at, submitted_at, started_at, ended_at,
                command, logs_path, exit_code, error_message,
                nodes, gpus, instance_type, cloud, region, zone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.experiment_id,
                job.cluster_name,
                job.sky_job_id,
                job.status.value,
                job.created_at.isoformat(),
                job.submitted_at.isoformat() if job.submitted_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.ended_at.isoformat() if job.ended_at else None,
                job.command,
                job.logs_path,
                job.exit_code,
                job.error_message,
                job.nodes,
                job.gpus,
                job.instance_type,
                job.cloud,
                job.region,
                job.zone,
            ),
        )
        await self._conn.commit()

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        cursor = await self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    async def get_jobs_for_experiment(self, experiment_id: str, limit: int = 10) -> list[Job]:
        """Get recent jobs for an experiment."""
        cursor = await self._conn.execute(
            """
            SELECT * FROM jobs
            WHERE experiment_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (experiment_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def get_running_jobs(self) -> list[Job]:
        """Get all active (PENDING or RUNNING) jobs."""
        cursor = await self._conn.execute(
            """
            SELECT * FROM jobs
            WHERE status IN ('PENDING', 'RUNNING')
            ORDER BY created_at DESC
            """
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def get_all_jobs(self, limit: int = 20, include_stopped: bool = False) -> list[Job]:
        """Get all jobs with optional filtering.

        Args:
            limit: Maximum number of jobs to return
            include_stopped: If True, include stopped/failed/succeeded jobs, otherwise only active jobs
        """
        if include_stopped:
            query = """
                SELECT * FROM jobs
                ORDER BY CAST(id AS INTEGER) DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT * FROM jobs
                WHERE status IN ('PENDING', 'RUNNING')
                ORDER BY CAST(id AS INTEGER) DESC
                LIMIT ?
            """

        cursor = await self._conn.execute(query, (limit,))
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        exit_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ):
        """Update job status and timestamps."""
        await self._conn.execute(
            """
            UPDATE jobs
            SET status = ?, started_at = ?, ended_at = ?, exit_code = ?, error_message = ?
            WHERE id = ?
            """,
            (
                status.value,
                started_at.isoformat() if started_at else None,
                ended_at.isoformat() if ended_at else None,
                exit_code,
                error_message,
                job_id,
            ),
        )
        await self._conn.commit()

    # Cluster operations

    async def save_cluster(self, cluster: Cluster):
        """Save or update cluster information."""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO clusters (
                name, status, num_nodes, instance_type, cloud, region, zone,
                created_at, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.name,
                cluster.status,
                cluster.num_nodes,
                cluster.instance_type,
                cluster.cloud,
                cluster.region,
                cluster.zone,
                cluster.created_at.isoformat() if cluster.created_at else None,
                cluster.last_seen.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_all_clusters(self) -> list[Cluster]:
        """Get all clusters."""
        cursor = await self._conn.execute("SELECT * FROM clusters ORDER BY last_seen DESC")
        rows = await cursor.fetchall()
        return [self._row_to_cluster(row) for row in rows]

    async def get_cluster(self, name: str) -> Optional[Cluster]:
        """Get cluster by name."""
        cursor = await self._conn.execute("SELECT * FROM clusters WHERE name = ?", (name,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_cluster(row)

    # Helper methods for row conversion

    def _row_to_experiment(self, row: aiosqlite.Row) -> Experiment:
        """Convert database row to Experiment model."""
        return Experiment(
            id=row["id"],
            name=row["name"],
            desired_state=DesiredState(row["desired_state"]),
            current_state=JobStatus(row["current_state"]),
            flags=json.loads(row["flags"]),
            base_command=row["base_command"],
            tool_path=row["tool_path"] if "tool_path" in row.keys() else None,
            run_name=row["run_name"],
            git_branch=row["git_branch"],
            current_job_id=row["current_job_id"],
            cluster_name=row["cluster_name"],
            nodes=row["nodes"],
            gpus=row["gpus"],
            instance_type=row["instance_type"],
            cloud=row["cloud"],
            region=row["region"],
            zone=row["zone"],
            spot=bool(row["spot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            wandb_link=row["wandb_link"],
            description=row["description"],
            tags=json.loads(row["tags"]),
            group=row["exp_group"],
            order=row["exp_order"],
            is_expanded=bool(row["is_expanded"]) if "is_expanded" in row.keys() else False,
        )

    def _row_to_job(self, row: aiosqlite.Row) -> Job:
        """Convert database row to Job model."""
        return Job(
            id=row["id"],
            experiment_id=row["experiment_id"],
            cluster_name=row["cluster_name"],
            sky_job_id=row["sky_job_id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            submitted_at=datetime.fromisoformat(row["submitted_at"]) if row["submitted_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            command=row["command"],
            logs_path=row["logs_path"],
            exit_code=row["exit_code"],
            error_message=row["error_message"],
            nodes=row["nodes"],
            gpus=row["gpus"],
            instance_type=row["instance_type"],
            cloud=row["cloud"],
            region=row["region"],
            zone=row["zone"],
        )

    def _row_to_cluster(self, row: aiosqlite.Row) -> Cluster:
        """Convert database row to Cluster model."""
        return Cluster(
            name=row["name"],
            status=row["status"],
            num_nodes=row["num_nodes"],
            instance_type=row["instance_type"],
            cloud=row["cloud"],
            region=row["region"],
            zone=row["zone"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            last_seen=datetime.fromisoformat(row["last_seen"]),
        )

    # User settings methods

    async def get_setting(self, key: str) -> Optional[dict]:
        """Get a user setting by key.

        Args:
            key: Setting key

        Returns:
            Setting value as dict, or None if not found
        """
        cursor = await self._conn.execute(
            "SELECT value FROM user_settings WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return json.loads(row["value"])

    async def set_setting(self, key: str, value: dict):
        """Set a user setting.

        Args:
            key: Setting key
            value: Setting value as dict (will be JSON serialized)
        """
        now = datetime.utcnow().isoformat()
        await self._conn.execute(
            """
            INSERT INTO user_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), now),
        )
        await self._conn.commit()

    # Checkpoint methods

    async def save_checkpoint(self, checkpoint: Checkpoint):
        """Save or update a checkpoint."""
        now = checkpoint.synced_at.isoformat()
        await self._conn.execute(
            """
            INSERT INTO checkpoints (
                experiment_id, epoch, model_path, replay_paths, metrics,
                created_at, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id, epoch) DO UPDATE SET
                model_path = excluded.model_path,
                replay_paths = excluded.replay_paths,
                metrics = excluded.metrics,
                synced_at = excluded.synced_at
            """,
            (
                checkpoint.experiment_id,
                checkpoint.epoch,
                checkpoint.model_path,
                json.dumps(checkpoint.replay_paths),
                json.dumps(checkpoint.metrics),
                checkpoint.created_at.isoformat(),
                now,
            ),
        )
        await self._conn.commit()

    async def get_checkpoints(self, experiment_id: str, limit: int = 50) -> list[Checkpoint]:
        """Get checkpoints for an experiment, ordered by epoch descending."""
        cursor = await self._conn.execute(
            """
            SELECT * FROM checkpoints
            WHERE experiment_id = ?
            ORDER BY epoch DESC
            LIMIT ?
            """,
            (experiment_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    async def get_checkpoint(self, experiment_id: str, epoch: int) -> Optional[Checkpoint]:
        """Get a specific checkpoint."""
        cursor = await self._conn.execute(
            """
            SELECT * FROM checkpoints
            WHERE experiment_id = ? AND epoch = ?
            """,
            (experiment_id, epoch),
        )
        row = await cursor.fetchone()
        return self._row_to_checkpoint(row) if row else None

    async def get_latest_epoch(self, experiment_id: str) -> Optional[int]:
        """Get the latest epoch number for an experiment."""
        cursor = await self._conn.execute(
            """
            SELECT MAX(epoch) as max_epoch FROM checkpoints
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )
        row = await cursor.fetchone()
        return row["max_epoch"] if row and row["max_epoch"] is not None else None

    def _row_to_checkpoint(self, row) -> Checkpoint:
        """Convert database row to Checkpoint object."""
        return Checkpoint(
            experiment_id=row["experiment_id"],
            epoch=row["epoch"],
            model_path=row["model_path"],
            replay_paths=json.loads(row["replay_paths"]),
            metrics=json.loads(row["metrics"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            synced_at=datetime.fromisoformat(row["synced_at"]),
        )
