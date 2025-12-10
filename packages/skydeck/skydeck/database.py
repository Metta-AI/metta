"""Database layer for SkyDeck dashboard using SQLite."""

import json
from datetime import datetime
from typing import Optional, Union

import aiosqlite

from .models import Checkpoint, Cluster, DesiredState, Experiment, ExperimentGroup, Job, JobStatus

# Type alias for experiment ID (int after migration, but accepting str for API compatibility)
ExperimentId = Union[int, str]


class Database:
    """Async SQLite database manager for SkyDeck."""

    def __init__(self, db_path: str | None = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file (default: ~/.skydeck/skydeck.db)
        """
        from pathlib import Path

        if db_path is None:
            db_path = str(Path.home() / ".skydeck" / "skydeck.db")
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
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

            -- Flag definitions table (extracted from Tool classes)
            CREATE TABLE IF NOT EXISTS flag_definitions (
                tool_path TEXT NOT NULL,
                flag TEXT NOT NULL,
                type TEXT NOT NULL,
                default_value TEXT,  -- JSON serialized
                required INTEGER NOT NULL DEFAULT 0,
                last_extracted TEXT NOT NULL,
                PRIMARY KEY (tool_path, flag)
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

            -- Experiment groups table
            CREATE TABLE IF NOT EXISTS experiment_groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                flags TEXT NOT NULL,  -- JSON array of flag names to display
                group_order INTEGER NOT NULL DEFAULT 0,
                collapsed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Experiment group membership (many-to-many)
            CREATE TABLE IF NOT EXISTS experiment_group_members (
                group_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                member_order INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (group_id, experiment_id),
                FOREIGN KEY (group_id) REFERENCES experiment_groups(id) ON DELETE CASCADE,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS operation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                experiment_id INTEGER,
                experiment_name TEXT,
                job_id TEXT,
                success INTEGER NOT NULL DEFAULT 1,
                error_message TEXT,
                output TEXT,
                user TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE SET NULL
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_jobs_experiment_id ON jobs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_experiments_desired_state ON experiments(desired_state);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_id ON checkpoints(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_group_members_experiment ON experiment_group_members(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_operation_logs_timestamp ON operation_logs(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_operation_logs_experiment_id ON operation_logs(experiment_id);
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

        # Migration: Add deleted column to experiments table if it doesn't exist
        if "deleted" not in column_names:
            await self._conn.execute("ALTER TABLE experiments ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
            await self._conn.commit()

        # Migration: Add observatory_url and policy_version columns to checkpoints table
        cursor = await self._conn.execute("PRAGMA table_info(checkpoints)")
        columns = await cursor.fetchall()
        checkpoint_columns = [col[1] for col in columns]

        if "observatory_url" not in checkpoint_columns:
            await self._conn.execute("ALTER TABLE checkpoints ADD COLUMN observatory_url TEXT")
            await self._conn.commit()

        if "policy_version" not in checkpoint_columns:
            await self._conn.execute("ALTER TABLE checkpoints ADD COLUMN policy_version TEXT")
            await self._conn.commit()

        if "policy_id" not in checkpoint_columns:
            await self._conn.execute("ALTER TABLE checkpoints ADD COLUMN policy_id TEXT")
            await self._conn.commit()

        if "policy_version_id" not in checkpoint_columns:
            await self._conn.execute("ALTER TABLE checkpoints ADD COLUMN policy_version_id TEXT")
            await self._conn.commit()

        # Migration: Change experiment.id from TEXT to INTEGER AUTOINCREMENT
        # Check if the migration is needed by looking at the column type
        cursor = await self._conn.execute("PRAGMA table_info(experiments)")
        columns = await cursor.fetchall()
        id_col = next((c for c in columns if c[1] == "id"), None)

        if id_col and id_col[2].upper() == "TEXT":
            # Need to migrate - experiment.id is still TEXT
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Migrating experiment.id from TEXT to INTEGER AUTOINCREMENT...")

            # Disable foreign key checks during migration
            await self._conn.execute("PRAGMA foreign_keys = OFF")

            # First, deduplicate names by appending suffix to duplicates
            cursor = await self._conn.execute("SELECT id, name FROM experiments ORDER BY created_at")
            experiments = await cursor.fetchall()

            # Track seen names and rename duplicates
            seen_names: set[str] = set()
            for exp_id, exp_name in experiments:
                if exp_name in seen_names:
                    # Find unique name by appending suffix
                    suffix = 2
                    new_name = f"{exp_name}_{suffix}"
                    while new_name in seen_names:
                        suffix += 1
                        new_name = f"{exp_name}_{suffix}"
                    # Update the duplicate name in the old table
                    await self._conn.execute("UPDATE experiments SET name = ? WHERE id = ?", (new_name, exp_id))
                    logger.info(f"Renamed duplicate experiment '{exp_name}' to '{new_name}'")
                    seen_names.add(new_name)
                else:
                    seen_names.add(exp_name)

            await self._conn.commit()

            # Re-fetch experiments after deduplication
            cursor = await self._conn.execute("SELECT id, name FROM experiments ORDER BY created_at")
            experiments = await cursor.fetchall()

            # Create mapping from old TEXT id to new INTEGER id
            id_map = {row[0]: i + 1 for i, row in enumerate(experiments)}

            # Create new experiments table with INTEGER id
            await self._conn.execute("""
                CREATE TABLE experiments_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    desired_state TEXT NOT NULL,
                    current_state TEXT NOT NULL,
                    flags TEXT NOT NULL,
                    base_command TEXT NOT NULL,
                    run_name TEXT,
                    git_branch TEXT,
                    current_job_id TEXT,
                    cluster_name TEXT,
                    nodes INTEGER NOT NULL DEFAULT 1,
                    gpus INTEGER NOT NULL DEFAULT 0,
                    instance_type TEXT,
                    cloud TEXT,
                    spot INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    wandb_link TEXT,
                    description TEXT,
                    tags TEXT NOT NULL,
                    exp_group TEXT,
                    exp_order INTEGER NOT NULL DEFAULT 0,
                    is_expanded INTEGER NOT NULL DEFAULT 0,
                    tool_path TEXT,
                    starred INTEGER NOT NULL DEFAULT 0,
                    deleted INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Copy experiments data with new INTEGER ids
            for old_id, new_id in id_map.items():
                await self._conn.execute(
                    """
                    INSERT INTO experiments_new (id, name, desired_state, current_state, flags, base_command,
                        run_name, git_branch, current_job_id, cluster_name, nodes, gpus, instance_type, cloud,
                        spot, created_at, updated_at, wandb_link, description, tags, exp_group, exp_order,
                        is_expanded, tool_path, starred, deleted)
                    SELECT ?, name, desired_state, current_state, flags, base_command,
                        run_name, git_branch, current_job_id, cluster_name, nodes, gpus, instance_type, cloud,
                        spot, created_at, updated_at, wandb_link, description, tags, exp_group, exp_order,
                        COALESCE(is_expanded, 0), tool_path, COALESCE(starred, 0), COALESCE(deleted, 0)
                    FROM experiments WHERE id = ?
                """,
                    (new_id, old_id),
                )

            # Update jobs table - experiment_id stays as the job name (not the experiment id)
            # This is correct since jobs.experiment_id stores the SkyPilot job name which matches experiment.name

            # Update checkpoints table - change experiment_id to integer
            await self._conn.execute("""
                CREATE TABLE checkpoints_new (
                    experiment_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    model_path TEXT,
                    replay_paths TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    observatory_url TEXT,
                    policy_version TEXT,
                    PRIMARY KEY (experiment_id, epoch),
                    FOREIGN KEY (experiment_id) REFERENCES experiments_new(id) ON DELETE CASCADE
                )
            """)

            for old_id, new_id in id_map.items():
                await self._conn.execute(
                    """
                    INSERT INTO checkpoints_new (experiment_id, epoch, model_path, replay_paths, metrics,
                        created_at, synced_at, observatory_url, policy_version)
                    SELECT ?, epoch, model_path, replay_paths, metrics, created_at, synced_at,
                        observatory_url, policy_version
                    FROM checkpoints WHERE experiment_id = ?
                """,
                    (new_id, old_id),
                )

            # Update experiment_group_members table
            await self._conn.execute("""
                CREATE TABLE experiment_group_members_new (
                    group_id TEXT NOT NULL,
                    experiment_id INTEGER NOT NULL,
                    member_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (group_id, experiment_id),
                    FOREIGN KEY (group_id) REFERENCES experiment_groups(id) ON DELETE CASCADE,
                    FOREIGN KEY (experiment_id) REFERENCES experiments_new(id) ON DELETE CASCADE
                )
            """)

            for old_id, new_id in id_map.items():
                await self._conn.execute(
                    """
                    INSERT INTO experiment_group_members_new (group_id, experiment_id, member_order)
                    SELECT group_id, ?, member_order
                    FROM experiment_group_members WHERE experiment_id = ?
                """,
                    (new_id, old_id),
                )

            # Drop old tables and rename new ones
            await self._conn.execute("DROP TABLE experiment_group_members")
            await self._conn.execute("DROP TABLE checkpoints")
            await self._conn.execute("DROP TABLE experiments")
            await self._conn.execute("ALTER TABLE experiments_new RENAME TO experiments")
            await self._conn.execute("ALTER TABLE checkpoints_new RENAME TO checkpoints")
            await self._conn.execute("ALTER TABLE experiment_group_members_new RENAME TO experiment_group_members")

            # Recreate indexes
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_desired_state ON experiments(desired_state)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_id ON checkpoints(experiment_id)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_members_experiment ON experiment_group_members(experiment_id)"
            )

            # Re-enable foreign keys
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.commit()

            logger.info(f"Migration complete. Migrated {len(id_map)} experiments to INTEGER ids.")

        # Migration: Add name_prefix column to experiment_groups table if it doesn't exist
        cursor = await self._conn.execute("PRAGMA table_info(experiment_groups)")
        columns = await cursor.fetchall()
        group_columns = [col[1] for col in columns]

        if "name_prefix" not in group_columns:
            await self._conn.execute("ALTER TABLE experiment_groups ADD COLUMN name_prefix TEXT")
            await self._conn.commit()

    # Experiment operations

    async def save_experiment(self, experiment: Experiment) -> Experiment:
        """Save or update an experiment. Returns the experiment with id populated."""
        if experiment.id is None:
            # Insert new experiment (let SQLite auto-generate id)
            cursor = await self._conn.execute(
                """
                INSERT INTO experiments (
                    name, desired_state, current_state, flags, base_command, tool_path, git_branch,
                    cluster_name, nodes, gpus, instance_type, cloud,
                    spot, created_at, updated_at, wandb_link, description, tags,
                    exp_group, exp_order, is_expanded, starred, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.name,
                    experiment.desired_state.value,
                    experiment.current_state.value,
                    json.dumps(experiment.flags),
                    experiment.base_command,
                    experiment.tool_path,
                    experiment.git_branch,
                    experiment.cluster_name,
                    experiment.nodes,
                    experiment.gpus,
                    experiment.instance_type,
                    experiment.cloud,
                    1 if experiment.spot else 0,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                    experiment.wandb_link,
                    experiment.description,
                    json.dumps(experiment.tags),
                    experiment.group,
                    experiment.order,
                    1 if experiment.is_expanded else 0,
                    1 if experiment.starred else 0,
                    1 if experiment.deleted else 0,
                ),
            )
            experiment.id = cursor.lastrowid
        else:
            # Update existing experiment
            await self._conn.execute(
                """
                UPDATE experiments SET
                    name = ?, desired_state = ?, current_state = ?, flags = ?, base_command = ?,
                    tool_path = ?, git_branch = ?, cluster_name = ?, nodes = ?, gpus = ?,
                    instance_type = ?, cloud = ?, spot = ?, updated_at = ?, wandb_link = ?,
                    description = ?, tags = ?, exp_group = ?, exp_order = ?, is_expanded = ?,
                    starred = ?, deleted = ?
                WHERE id = ?
                """,
                (
                    experiment.name,
                    experiment.desired_state.value,
                    experiment.current_state.value,
                    json.dumps(experiment.flags),
                    experiment.base_command,
                    experiment.tool_path,
                    experiment.git_branch,
                    experiment.cluster_name,
                    experiment.nodes,
                    experiment.gpus,
                    experiment.instance_type,
                    experiment.cloud,
                    1 if experiment.spot else 0,
                    experiment.updated_at.isoformat(),
                    experiment.wandb_link,
                    experiment.description,
                    json.dumps(experiment.tags),
                    experiment.group,
                    experiment.order,
                    1 if experiment.is_expanded else 0,
                    1 if experiment.starred else 0,
                    1 if experiment.deleted else 0,
                    experiment.id,
                ),
            )
        await self._conn.commit()
        return experiment

    async def get_experiment(self, experiment_id: ExperimentId, include_deleted: bool = False) -> Optional[Experiment]:
        """Get experiment by ID.

        Args:
            experiment_id: The experiment ID to look up (int, accepts str for API compatibility)
            include_deleted: If True, also return soft-deleted experiments
        """
        if include_deleted:
            cursor = await self._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM experiments WHERE id = ? AND deleted = 0", (experiment_id,)
            )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    async def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name (among non-deleted experiments only).

        Args:
            name: The experiment name to look up
        """
        cursor = await self._conn.execute("SELECT * FROM experiments WHERE name = ? AND deleted = 0", (name,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    async def get_all_experiments(self) -> list[Experiment]:
        """Get all experiments (excluding deleted ones)."""
        cursor = await self._conn.execute(
            "SELECT * FROM experiments WHERE deleted = 0 ORDER BY exp_order ASC, created_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_experiment(row) for row in rows]

    async def delete_experiment(self, experiment_id: ExperimentId):
        """Soft-delete experiment by marking it as deleted."""
        await self._conn.execute("UPDATE experiments SET deleted = 1 WHERE id = ?", (experiment_id,))
        await self._conn.commit()

    async def undelete_experiment(self, experiment_id: ExperimentId):
        """Restore a soft-deleted experiment."""
        await self._conn.execute("UPDATE experiments SET deleted = 0 WHERE id = ?", (experiment_id,))
        await self._conn.commit()

    async def update_experiment_state(self, experiment_id: ExperimentId, current_state: JobStatus):
        """Update experiment current state."""
        await self._conn.execute(
            """
            UPDATE experiments
            SET current_state = ?, updated_at = ?
            WHERE id = ?
            """,
            (current_state.value, datetime.utcnow().isoformat(), experiment_id),
        )
        await self._conn.commit()

    async def update_experiment_desired_state(self, experiment_id: ExperimentId, desired_state: DesiredState):
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

    async def update_experiment_flags(self, experiment_id: ExperimentId, flags: dict):
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

    async def set_experiment_cluster(self, experiment_id: ExperimentId, cluster_name: str):
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

    async def update_experiment_expanded(self, experiment_id: ExperimentId, is_expanded: bool):
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
                nodes, gpus, instance_type, cloud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    async def get_current_job_for_experiment(self, experiment_id: str) -> Optional[Job]:
        """Get the currently active (RUNNING or PENDING) job for an experiment.

        Returns the most recently created active job, or None if no active jobs exist.
        """
        cursor = await self._conn.execute(
            """
            SELECT * FROM jobs
            WHERE experiment_id = ? AND status IN ('PENDING', 'RUNNING')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (experiment_id,),
        )
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
                name, status, num_nodes, instance_type, cloud,
                created_at, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.name,
                cluster.status,
                cluster.num_nodes,
                cluster.instance_type,
                cluster.cloud,
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
            git_branch=row["git_branch"],
            cluster_name=row["cluster_name"],
            nodes=row["nodes"],
            gpus=row["gpus"],
            instance_type=row["instance_type"],
            cloud=row["cloud"],
            spot=bool(row["spot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            wandb_link=row["wandb_link"],
            description=row["description"],
            tags=json.loads(row["tags"]),
            group=row["exp_group"],
            order=row["exp_order"],
            is_expanded=bool(row["is_expanded"]) if "is_expanded" in row.keys() else False,
            starred=bool(row["starred"]) if "starred" in row.keys() else False,
            deleted=bool(row["deleted"]) if "deleted" in row.keys() else False,
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
        )

    def _row_to_cluster(self, row: aiosqlite.Row) -> Cluster:
        """Convert database row to Cluster model."""
        return Cluster(
            name=row["name"],
            status=row["status"],
            num_nodes=row["num_nodes"],
            instance_type=row["instance_type"],
            cloud=row["cloud"],
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
                created_at, synced_at, observatory_url, policy_version, policy_id, policy_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id, epoch) DO UPDATE SET
                model_path = excluded.model_path,
                replay_paths = excluded.replay_paths,
                metrics = excluded.metrics,
                synced_at = excluded.synced_at,
                observatory_url = excluded.observatory_url,
                policy_version = excluded.policy_version,
                policy_id = excluded.policy_id,
                policy_version_id = excluded.policy_version_id
            """,
            (
                checkpoint.experiment_id,
                checkpoint.epoch,
                checkpoint.model_path,
                json.dumps(checkpoint.replay_paths),
                json.dumps(checkpoint.metrics),
                checkpoint.created_at.isoformat(),
                now,
                checkpoint.observatory_url,
                checkpoint.policy_version,
                checkpoint.policy_id,
                checkpoint.policy_version_id,
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
            observatory_url=row["observatory_url"] if "observatory_url" in row.keys() else None,
            policy_version=row["policy_version"] if "policy_version" in row.keys() else None,
            policy_id=row["policy_id"] if "policy_id" in row.keys() else None,
            policy_version_id=row["policy_version_id"] if "policy_version_id" in row.keys() else None,
        )

    # Operation log methods

    async def save_operation_log(self, log: "OperationLog"):
        """Save an operation log entry."""

        await self._conn.execute(
            """
            INSERT INTO operation_logs (
                timestamp, operation_type, experiment_id, experiment_name,
                job_id, success, error_message, output, user
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.timestamp.isoformat(),
                log.operation_type.value,
                log.experiment_id,
                log.experiment_name,
                log.job_id,
                1 if log.success else 0,
                log.error_message,
                log.output,
                log.user,
            ),
        )
        await self._conn.commit()

    async def get_operation_logs(self, limit: int = 100) -> list["OperationLog"]:
        """Get recent operation logs."""

        cursor = await self._conn.execute(
            """
            SELECT * FROM operation_logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_operation_log(row) for row in rows]

    def _row_to_operation_log(self, row) -> "OperationLog":
        """Convert database row to OperationLog object."""
        from .models import OperationLog, OperationType

        return OperationLog(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            operation_type=OperationType(row["operation_type"]),
            experiment_id=row["experiment_id"],
            experiment_name=row["experiment_name"],
            job_id=row["job_id"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            output=row["output"] if "output" in row.keys() else None,
            user=row["user"],
        )

    # Experiment group methods

    async def save_group(self, group: ExperimentGroup):
        """Save or update an experiment group."""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO experiment_groups (
                id, name, name_prefix, flags, group_order, collapsed, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group.id,
                group.name,
                group.name_prefix,
                json.dumps(group.flags),
                group.order,
                1 if group.collapsed else 0,
                group.created_at.isoformat(),
                group.updated_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_group(self, group_id: str) -> Optional[ExperimentGroup]:
        """Get a group by ID."""
        cursor = await self._conn.execute("SELECT * FROM experiment_groups WHERE id = ?", (group_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_group(row)

    async def get_all_groups(self) -> list[ExperimentGroup]:
        """Get all groups ordered by their display order."""
        cursor = await self._conn.execute("SELECT * FROM experiment_groups ORDER BY group_order ASC")
        rows = await cursor.fetchall()
        return [self._row_to_group(row) for row in rows]

    async def delete_group(self, group_id: str):
        """Delete a group and its memberships."""
        await self._conn.execute("DELETE FROM experiment_group_members WHERE group_id = ?", (group_id,))
        await self._conn.execute("DELETE FROM experiment_groups WHERE id = ?", (group_id,))
        await self._conn.commit()

    async def reorder_groups(self, group_ids: list[str]):
        """Update the order of groups."""
        for index, group_id in enumerate(group_ids):
            await self._conn.execute(
                "UPDATE experiment_groups SET group_order = ? WHERE id = ?",
                (index, group_id),
            )
        await self._conn.commit()

    def _row_to_group(self, row) -> ExperimentGroup:
        """Convert database row to ExperimentGroup object."""
        # Handle name_prefix field which may not exist in older databases
        name_prefix = None
        try:
            name_prefix = row["name_prefix"]
        except (KeyError, IndexError):
            pass

        return ExperimentGroup(
            id=row["id"],
            name=row["name"],
            name_prefix=name_prefix,
            flags=json.loads(row["flags"]),
            order=row["group_order"],
            collapsed=bool(row["collapsed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # Group membership methods

    async def add_experiment_to_group(
        self, group_id: str, experiment_id: str, order: int = 0, multi_home: bool = False
    ):
        """Add an experiment to a group.

        Args:
            group_id: Target group ID
            experiment_id: Experiment ID to add
            order: Order within the group
            multi_home: If False, remove from all other groups first
        """
        if not multi_home:
            # Remove from all other groups
            await self._conn.execute(
                "DELETE FROM experiment_group_members WHERE experiment_id = ? AND group_id != ?",
                (experiment_id, group_id),
            )

        # Add to this group (or update order if already exists)
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO experiment_group_members (group_id, experiment_id, member_order)
            VALUES (?, ?, ?)
            """,
            (group_id, experiment_id, order),
        )
        await self._conn.commit()

    async def remove_experiment_from_group(self, group_id: str, experiment_id: str):
        """Remove an experiment from a group."""
        await self._conn.execute(
            "DELETE FROM experiment_group_members WHERE group_id = ? AND experiment_id = ?",
            (group_id, experiment_id),
        )
        await self._conn.commit()

    async def get_experiments_in_group(self, group_id: str) -> list[Experiment]:
        """Get all experiments in a group, ordered by their position."""
        cursor = await self._conn.execute(
            """
            SELECT e.* FROM experiments e
            JOIN experiment_group_members m ON e.id = m.experiment_id
            WHERE m.group_id = ? AND e.deleted = 0
            ORDER BY m.member_order ASC
            """,
            (group_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_experiment(row) for row in rows]

    async def get_ungrouped_experiments(self) -> list[Experiment]:
        """Get all experiments not in any group."""
        cursor = await self._conn.execute(
            """
            SELECT e.* FROM experiments e
            WHERE e.deleted = 0 AND e.id NOT IN (
                SELECT experiment_id FROM experiment_group_members
            )
            ORDER BY e.exp_order ASC, e.created_at DESC
            """
        )
        rows = await cursor.fetchall()
        return [self._row_to_experiment(row) for row in rows]

    async def get_groups_for_experiment(self, experiment_id: str) -> list[str]:
        """Get all group IDs that an experiment belongs to."""
        cursor = await self._conn.execute(
            "SELECT group_id FROM experiment_group_members WHERE experiment_id = ?",
            (experiment_id,),
        )
        rows = await cursor.fetchall()
        return [row["group_id"] for row in rows]

    async def reorder_experiments_in_group(self, group_id: str, experiment_ids: list[str]):
        """Update the order of experiments within a group."""
        for index, exp_id in enumerate(experiment_ids):
            await self._conn.execute(
                "UPDATE experiment_group_members SET member_order = ? WHERE group_id = ? AND experiment_id = ?",
                (index, group_id, exp_id),
            )
        await self._conn.commit()

    # Flag definition methods

    async def save_flag_definitions(self, tool_path: str, flags: list[dict]):
        """Save or update flag definitions for a tool path.

        Args:
            tool_path: Tool path (e.g., "arena.train")
            flags: List of flag definitions from extract_flags_from_tool_path()
        """
        now = datetime.utcnow().isoformat()
        # Delete old flags for this tool path
        await self._conn.execute("DELETE FROM flag_definitions WHERE tool_path = ?", (tool_path,))
        # Insert new flags
        for flag_def in flags:
            await self._conn.execute(
                """
                INSERT INTO flag_definitions (tool_path, flag, type, default_value, required, last_extracted)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_path,
                    flag_def["flag"],
                    flag_def["type"],
                    json.dumps(flag_def["default"]) if flag_def["default"] is not None else None,
                    1 if flag_def["required"] else 0,
                    now,
                ),
            )
        await self._conn.commit()

    async def get_flag_definitions(self, tool_path: str) -> list[dict]:
        """Get flag definitions for a tool path.

        Args:
            tool_path: Tool path (e.g., "arena.train")

        Returns:
            List of flag definitions with keys: flag, type, default, required
        """
        cursor = await self._conn.execute(
            "SELECT * FROM flag_definitions WHERE tool_path = ?",
            (tool_path,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "flag": row["flag"],
                "type": row["type"],
                "default": json.loads(row["default_value"]) if row["default_value"] else None,
                "required": bool(row["required"]),
            }
            for row in rows
        ]

    async def get_all_unique_flags(self) -> list[dict]:
        """Get all unique flag names across all tool paths.

        Returns:
            List of unique flags with their most common type and default
        """
        cursor = await self._conn.execute(
            """
            SELECT flag, type, default_value, required, COUNT(*) as count
            FROM flag_definitions
            GROUP BY flag
            ORDER BY flag ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "flag": row["flag"],
                "type": row["type"],
                "default": json.loads(row["default_value"]) if row["default_value"] else None,
                "required": bool(row["required"]),
            }
            for row in rows
        ]
