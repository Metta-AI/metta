import hashlib
import logging
import secrets
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Literal

from psycopg import Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import BaseModel, Field

from metta.app_backend.query_logger import execute_single_row_query_and_log
from metta.app_backend.migrations import MIGRATIONS
from metta.app_backend.schema_manager import run_migrations

TaskStatus = Literal["unprocessed", "canceled", "done", "error"]


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    clear_assignee: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


class TrainingRunRow(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    user_id: str
    finished_at: datetime | None
    status: str
    url: str | None
    description: str | None
    tags: list[str]


class MachineTokenRow(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    expiration_time: datetime
    last_used_at: datetime | None


class EvalTaskRow(BaseModel):
    """Row model that matches the eval_tasks table structure."""

    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: str
    assigned_at: datetime | None
    assignee: str | None
    created_at: datetime
    attributes: dict[str, Any]
    retries: int
    user_id: str | None
    updated_at: datetime


class EvalTaskWithPolicyName(BaseModel):
    """Extended eval task row that includes policy name from JOIN with policies table."""

    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: str
    assigned_at: datetime | None
    assignee: str | None
    created_at: datetime
    attributes: dict[str, Any]
    retries: int
    policy_name: str
    policy_url: str
    user_id: str | None
    updated_at: datetime


class SweepRow(BaseModel):
    id: uuid.UUID
    name: str
    project: str
    entity: str
    wandb_sweep_id: str
    state: str
    run_counter: int
    user_id: str
    created_at: datetime
    updated_at: datetime


class PolicyRow(BaseModel):
    id: uuid.UUID
    name: str
    url: str | None


class PolicyEval(BaseModel):
    num_agents: int
    total_score: float


class CoGamesSubmissionRow(BaseModel):
    id: uuid.UUID
    user_id: str
    name: str | None
    s3_path: str
    created_at: datetime


logger = logging.getLogger(name="metta_repo")


class MettaRepo:
    def __init__(self, db_uri: str) -> None:
        self.db_uri = db_uri
        self._pool: AsyncConnectionPool | None = None
        # Run migrations synchronously during initialization
        with Connection.connect(self.db_uri) as con:
            run_migrations(con, MIGRATIONS)

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(self.db_uri, min_size=2, max_size=20, open=False)
            await self._pool.open()
        return self._pool

    @asynccontextmanager
    async def connect(self):
        pool = await self._ensure_pool()
        try:
            async with pool.connection(timeout=5) as conn:
                yield conn
        except PoolTimeout as e:
            stats = pool.get_stats()
            logger.error(f"Error connecting to database: {e}. Pool stats: {stats}", exc_info=True)

            await pool.check()
            async with pool.connection() as conn:
                yield conn

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.close()
            except RuntimeError:
                # Event loop might be closed, ignore
                pass

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    # All methods are async - no sync versions

    async def get_policy_by_id(self, policy_id: uuid.UUID) -> PolicyRow | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(PolicyRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, url
                    FROM policies
                    WHERE id = %s
                    """,
                    (policy_id,),
                )
                return await cur.fetchone()

    async def get_policy_ids(self, policy_names: list[str]) -> dict[str, uuid.UUID]:
        if not policy_names:
            return {}

        async with self.connect() as con:
            res = await con.execute(
                """
                SELECT id, name FROM policies WHERE name = ANY(%s)
                """,
                (policy_names,),
            )
            rows = await res.fetchall()
            return {row[1]: row[0] for row in rows}

    async def create_training_run(
        self,
        name: str,
        user_id: str,
        attributes: dict[str, str],
        url: str | None,
        description: str | None,
        tags: list[str] | None,
    ) -> uuid.UUID:
        status = "running"
        async with self.connect() as con:
            # Try to insert a new training run, but if it already exists, return the existing ID
            result = await con.execute(
                """
                INSERT INTO training_runs (name, user_id, attributes, status, url, description, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, name) DO NOTHING
                RETURNING id
                """,
                (name, user_id, Jsonb(attributes), status, url, description, tags),
            )
            row = await result.fetchone()
            if row is None:
                # If no result, the run already exists, so fetch its ID
                result = await con.execute(
                    """
                    SELECT id FROM training_runs WHERE user_id = %s AND name = %s
                    """,
                    (user_id, name),
                )
                row = await result.fetchone()
                if row is None:
                    raise RuntimeError("Failed to find existing training run")
            return row[0]

    async def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE training_runs
                SET status = %s, finished_at = CASE WHEN %s != 'running' THEN CURRENT_TIMESTAMP ELSE finished_at END
                WHERE id = %s
                """,
                (status, status, run_id),
            )
            if result.rowcount == 0:
                raise ValueError(f"Training run with ID {run_id} not found")

    async def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, Any],
    ) -> uuid.UUID:
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO epochs (run_id, start_training_epoch, end_training_epoch, attributes)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, start_training_epoch, end_training_epoch, Jsonb(attributes)),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert policy epoch")
            return row[0]

    async def create_policy(
        self,
        name: str,
        description: str | None,
        url: str | None,
        epoch_id: uuid.UUID | None,
    ) -> uuid.UUID:
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO policies (name, description, url, epoch_id) VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, description, url, epoch_id),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert policy")
            return row[0]

    async def record_episode(
        self,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        eval_name: str,
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None,
        replay_url: str | None,
        attributes: dict[str, Any],
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> uuid.UUID:
        async with self.connect() as con:
            # Validate that eval name is in the format of 'eval_category/env_name'
            parts = eval_name.split("/")
            if len(parts) != 2:
                raise ValueError("Eval name must be in the format of 'eval_category/env_name'")
            eval_category, env_name = parts

            # Insert into episodes table
            result = await con.execute(
                """
                INSERT INTO episodes (
                    replay_url,
                    eval_name,
                    eval_category,
                    env_name,
                    primary_policy_id,
                    stats_epoch,
                    attributes,
                    eval_task_id,
                    thumbnail_url
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id, internal_id
                """,
                (
                    replay_url,
                    eval_name,
                    eval_category,
                    env_name,
                    primary_policy_id,
                    stats_epoch,
                    Jsonb(attributes),
                    eval_task_id,
                    thumbnail_url,
                ),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert episode record")
            episode_id = row[0]
            episode_internal_id = row[1]

            # Insert agent policies
            for agent_id, policy_id in agent_policies.items():
                await con.execute(
                    """
                    INSERT INTO episode_agent_policies (
                        episode_id,
                        policy_id,
                        agent_id
                    ) VALUES (%s, %s, %s)
                    """,
                    (episode_id, policy_id, agent_id),
                )

            # Insert agent metrics in bulk
            rows: list[tuple[int, int, str, float]] = []
            for agent_id, metrics in agent_metrics.items():
                for metric_name, value in metrics.items():
                    rows.append((episode_internal_id, agent_id, metric_name, value))

            async with con.cursor() as cursor:
                await cursor.executemany(
                    """
                  INSERT INTO episode_agent_metrics (episode_internal_id, agent_id, metric, value)
                  VALUES (%s, %s, %s, %s)
                  """,
                    rows,
                )

            # Add tags if provided
            if tags:
                tag_rows = [(episode_id, tag) for tag in tags]
                async with con.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO episode_tags (episode_id, tag)
                        VALUES (%s, %s)
                        ON CONFLICT (episode_id, tag) DO NOTHING
                        """,
                        tag_rows,
                    )

            return episode_id

    async def get_training_runs(self) -> list[TrainingRunRow]:
        """Get all training runs."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(TrainingRunRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, created_at, user_id, finished_at, status, url, description,
                           COALESCE(tags, ARRAY[]::TEXT[]) as tags
                    FROM training_runs
                    ORDER BY created_at DESC
                    """
                )
                return await cur.fetchall()

    async def get_training_run(self, run_id: str) -> TrainingRunRow | None:
        """Get a specific training run by ID."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return None

        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(TrainingRunRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, created_at, user_id, finished_at, status, url, description,
                           COALESCE(tags, ARRAY[]::TEXT[]) as tags
                    FROM training_runs
                    WHERE id = %s
                    """,
                    (run_uuid,),
                )
                return await cur.fetchone()

    async def create_machine_token(self, user_id: str, name: str, expiration_days: int = 365) -> str:
        """Create a new machine token for a user."""
        # Generate a secure random token
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token)

        # Set expiration time
        expiration_time = datetime.now() + timedelta(days=expiration_days)

        async with self.connect() as con:
            await con.execute(
                """
                INSERT INTO machine_tokens (user_id, name, token_hash, expiration_time)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, name, token_hash, expiration_time),
            )

        return token

    async def list_machine_tokens(self, user_id: str) -> list[MachineTokenRow]:
        """List all machine tokens for a user."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(MachineTokenRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, created_at, expiration_time, last_used_at
                    FROM machine_tokens
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                return await cur.fetchall()

    async def delete_machine_token(self, user_id: str, token_id: str) -> bool:
        """Delete a machine token."""
        try:
            token_uuid = uuid.UUID(token_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                DELETE FROM machine_tokens
                WHERE id = %s AND user_id = %s
                """,
                (token_uuid, user_id),
            )
            return result.rowcount > 0

    async def validate_machine_token(self, token: str) -> str | None:
        """Validate a machine token and return the user_id if valid."""
        token_hash = self._hash_token(token)

        async with self.connect() as con:
            query = """
                UPDATE machine_tokens
                SET last_used_at = CURRENT_TIMESTAMP
                WHERE token_hash = %s AND expiration_time > CURRENT_TIMESTAMP
                RETURNING user_id
                """
            result = await execute_single_row_query_and_log(
                con,
                query,
                (token_hash,),
                "validate_machine_token",
            )

            if result:
                return result[0]
            return None

    async def update_training_run_description(self, user_id: str, run_id: str, description: str) -> bool:
        """Update the description of a training run."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE training_runs
                SET description = %s
                WHERE id = %s AND user_id = %s
                """,
                (description, run_uuid, user_id),
            )
            return result.rowcount > 0

    async def update_training_run_tags(self, user_id: str, run_id: str, tags: list[str]) -> bool:
        """Update the tags of a training run."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE training_runs
                SET tags = %s
                WHERE id = %s AND user_id = %s
                """,
                (tags, run_uuid, user_id),
            )
            return result.rowcount > 0

    async def create_eval_task(
        self,
        policy_id: uuid.UUID,
        sim_suite: str,
        attributes: dict[str, Any],
        user_id: str | None = None,
    ) -> EvalTaskRow:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute(
                    """
                    INSERT INTO eval_tasks (policy_id, sim_suite, attributes, user_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, policy_id, sim_suite, status, assigned_at,
                             assignee, created_at, attributes, retries, user_id, updated_at
                    """,
                    (policy_id, sim_suite, Jsonb(attributes), user_id),
                )
                row = await cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to create eval task")
                return row

    async def get_available_tasks(self, limit: int = 200) -> list[EvalTaskWithPolicyName]:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                await cur.execute(
                    """
                    SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                           et.assignee, et.created_at, et.attributes, et.retries,
                           p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                    FROM eval_tasks et
                    JOIN policies p ON et.policy_id = p.id
                    WHERE status = 'unprocessed'
                      AND assignee IS NULL
                    ORDER BY et.created_at ASC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return await cur.fetchall()

    async def claim_tasks(
        self,
        task_ids: list[uuid.UUID],
        assignee: str,
    ) -> list[uuid.UUID]:
        if not task_ids:
            return []

        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE eval_tasks
                SET assignee = %s, assigned_at = NOW(), retries = retries +1, updated_at = NOW()
                WHERE id = ANY(%s)
                    AND status = 'unprocessed'
                    AND assignee IS NULL
                RETURNING id
                """,
                (assignee, task_ids),
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_claimed_tasks(self, assignee: str | None = None) -> list[EvalTaskWithPolicyName]:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                if assignee is not None:
                    await cur.execute(
                        """
                        SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                                et.assignee, et.created_at, et.attributes, et.retries,
                                p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                        FROM eval_tasks et
                        JOIN policies p ON et.policy_id = p.id
                        WHERE assignee = %s AND status = 'unprocessed'
                        ORDER BY et.created_at ASC
                        """,
                        (assignee,),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                                et.assignee, et.created_at, et.attributes, et.retries,
                                p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                        FROM eval_tasks et
                        JOIN policies p ON et.policy_id = p.id
                        WHERE status = 'unprocessed' AND assignee IS NOT NULL
                        ORDER BY et.created_at ASC
                        """,
                    )
                return await cur.fetchall()

    async def update_task_statuses(
        self,
        updates: dict[uuid.UUID, TaskStatusUpdate],
        require_assignee: str | None = None,
    ) -> dict[uuid.UUID, TaskStatus]:
        if not updates:
            return {}

        updated = {}
        async with self.connect() as con:
            for task_id, update in updates.items():
                if require_assignee:
                    filter_clause = "id = %s AND assignee = %s"
                    filter_params = (task_id, require_assignee)
                else:
                    filter_clause = "id = %s"
                    filter_params = (task_id,)

                if update.clear_assignee:
                    assignee_clause = "assignee = NULL,"
                else:
                    assignee_clause = ""

                query = f"""
                UPDATE eval_tasks
                SET status = %s,
                    {assignee_clause}
                    attributes = COALESCE(attributes, '{{}}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE {filter_clause}
                RETURNING id
                """
                params = (update.status, Jsonb(update.attributes)) + filter_params
                result = await con.execute(query, params)

                if result.rowcount > 0:
                    updated[task_id] = update.status

        return updated

    async def count_tasks(self, where_clause: str) -> int:
        async with self.connect() as con:
            result = await con.execute(
                f"SELECT COUNT(*) FROM eval_tasks WHERE {where_clause}",  # type: ignore
            )
            res = await result.fetchone()
            if res is None:
                raise RuntimeError(f"Failed to count tasks with where clause {where_clause}")
            return res[0]

    async def get_avg_runtime(self, where_clause: str) -> float | None:
        async with self.connect() as con:
            result = await con.execute(
                f"SELECT EXTRACT(EPOCH FROM AVG(updated_at - assigned_at)) FROM eval_tasks WHERE {where_clause}",  # type: ignore
            )
            res = await result.fetchone()
            if res is None:
                raise RuntimeError(f"Failed to get average runtime with where clause {where_clause}")
            return res[0]

    async def add_episode_tags(self, episode_ids: list[uuid.UUID], tag: str) -> int:
        """Add a tag to multiple episodes by UUID. Returns number of episodes tagged."""
        if not episode_ids:
            return 0

        async with self.connect() as con:
            # Use INSERT ... ON CONFLICT DO NOTHING to avoid duplicate key errors
            rows_affected = 0
            for episode_id in episode_ids:
                result = await con.execute(
                    """
                    INSERT INTO episode_tags (episode_id, tag)
                    VALUES (%s, %s)
                    ON CONFLICT (episode_id, tag) DO NOTHING
                    """,
                    (episode_id, tag),
                )
                rows_affected += result.rowcount
            return rows_affected

    async def remove_episode_tags(self, episode_ids: list[uuid.UUID], tag: str) -> int:
        """Remove a tag from multiple episodes by UUID. Returns number of episodes untagged."""
        if not episode_ids:
            return 0

        async with self.connect() as con:
            result = await con.execute(
                """
                DELETE FROM episode_tags
                WHERE episode_id = ANY(%s) AND tag = %s
                """,
                (episode_ids, tag),
            )
            return result.rowcount

    async def get_episode_tags(self, episode_ids: list[uuid.UUID]) -> dict[str, list[str]]:
        """Get all tags for the given episode UUIDs."""
        if not episode_ids:
            return {}

        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT episode_id, tag
                FROM episode_tags
                WHERE episode_id = ANY(%s)
                ORDER BY episode_id, tag
                """,
                (episode_ids,),
            )
            rows = await result.fetchall()

            # Group tags by episode UUID (converted to string for JSON serialization)
            tags_by_episode = {}
            for episode_uuid, tag in rows:
                episode_key = str(episode_uuid)
                if episode_key not in tags_by_episode:
                    tags_by_episode[episode_key] = []
                tags_by_episode[episode_key].append(tag)

            return tags_by_episode

    async def get_all_episode_tags(self) -> list[str]:
        """Get all distinct tags that exist in the system."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT DISTINCT tag
                FROM episode_tags
                ORDER BY tag
                """
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def create_sweep(self, name: str, project: str, entity: str, wandb_sweep_id: str, user_id: str) -> uuid.UUID:
        """Create a new sweep."""
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO sweeps (name, project, entity, wandb_sweep_id, user_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, project, entity, wandb_sweep_id, user_id),
            )
            row = await result.fetchone()
            if row is None:
                raise ValueError("Failed to create sweep")
            return row[0]

    async def get_sweep_by_name(self, name: str) -> SweepRow | None:
        """Get sweep by name."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(SweepRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, project, entity, wandb_sweep_id, state, run_counter,
                           user_id, created_at, updated_at
                    FROM sweeps
                    WHERE name = %s
                    """,
                    (name,),
                )
                return await cur.fetchone()

    async def get_next_sweep_run_counter(self, sweep_id: uuid.UUID) -> int:
        """Atomically increment and return the next run counter for a sweep."""
        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE sweeps
                SET run_counter = run_counter + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING run_counter
                """,
                (sweep_id,),
            )
            row = await result.fetchone()
            if row is None:
                raise ValueError(f"Sweep {sweep_id} not found")
            return row[0]

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> EvalTaskWithPolicyName | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                await cur.execute(
                    """
                    SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                           et.assignee, et.created_at, et.attributes, et.retries,
                           p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                    FROM eval_tasks et
                    JOIN policies p ON et.policy_id = p.id
                    WHERE assignee = %s
                      AND assigned_at IS NOT NULL
                    ORDER BY assigned_at DESC
                    LIMIT 1
                    """,
                    (assignee,),
                )
                return await cur.fetchone()

    async def get_task_by_id(self, task_id: uuid.UUID) -> EvalTaskWithPolicyName | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                await cur.execute(
                    """
                    SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                           et.assignee, et.created_at, et.attributes, et.retries,
                           p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                    FROM eval_tasks et
                    JOIN policies p ON et.policy_id = p.id
                    WHERE et.id = %s
                    """,
                    (task_id,),
                )
                return await cur.fetchone()

    async def get_all_tasks(
        self,
        limit: int = 500,
        statuses: list[TaskStatus] | None = None,
        git_hash: str | None = None,
        policy_ids: list[uuid.UUID] | None = None,
        sim_suites: list[str] | None = None,
    ) -> list[EvalTaskWithPolicyName]:
        async with self.connect() as con:
            # Build the WHERE clause dynamically
            where_conditions = []
            params = []

            if statuses:
                placeholders = ", ".join(["%s"] * len(statuses))
                where_conditions.append(f"et.status IN ({placeholders})")
                params.extend(statuses)

            if git_hash:
                where_conditions.append("et.attributes->>'git_hash' = %s")
                params.append(git_hash)

            if policy_ids:
                placeholders = ", ".join(["%s"] * len(policy_ids))
                where_conditions.append(f"et.policy_id IN ({placeholders})")
                params.extend(policy_ids)

            if sim_suites:
                placeholders = ", ".join(["%s"] * len(sim_suites))
                where_conditions.append(f"et.sim_suite IN ({placeholders})")
                params.extend(sim_suites)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            params.append(limit)

            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                await cur.execute(
                    f"""
                    SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                           et.assignee, et.created_at, et.attributes, et.retries,
                           p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                    FROM eval_tasks et
                    LEFT JOIN policies p ON et.policy_id = p.id
                    WHERE {where_clause}
                    ORDER BY et.created_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                return await cur.fetchall()

    async def get_tasks_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        policy_name: str | None = None,
        sim_suite: str | None = None,
        status: str | None = None,
        assignee: str | None = None,
        user_id: str | None = None,
        retries: str | None = None,
        created_at: str | None = None,
        assigned_at: str | None = None,
        updated_at: str | None = None,
        include_attributes: bool = False,
    ) -> tuple[list[EvalTaskWithPolicyName], int]:
        async with self.connect() as con:
            where_conditions = []
            params = []

            # Add text-based filters using ILIKE for case-insensitive substring search
            if policy_name:
                where_conditions.append("p.name ILIKE %s")
                params.append(f"%{policy_name}%")

            if sim_suite:
                where_conditions.append("et.sim_suite ILIKE %s")
                params.append(f"%{sim_suite}%")

            if status:
                # Use exact match for status since it's an enum-like field
                where_conditions.append("et.status = %s")
                params.append(status)

            if assignee:
                where_conditions.append("et.assignee ILIKE %s")
                params.append(f"%{assignee}%")

            if user_id:
                where_conditions.append("et.user_id ILIKE %s")
                params.append(f"%{user_id}%")

            if retries:
                where_conditions.append("CAST(et.retries AS TEXT) ILIKE %s")
                params.append(f"%{retries}%")

            if created_at:
                where_conditions.append("CAST(et.created_at AS TEXT) ILIKE %s")
                params.append(f"%{created_at}%")

            if assigned_at:
                where_conditions.append("CAST(et.assigned_at AS TEXT) ILIKE %s")
                params.append(f"%{assigned_at}%")

            if updated_at:
                where_conditions.append("CAST(et.updated_at AS TEXT) ILIKE %s")
                params.append(f"%{updated_at}%")

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # Get total count
            count_query = f"""
                SELECT COUNT(*)
                FROM eval_tasks et
                LEFT JOIN policies p ON et.policy_id = p.id
                WHERE {where_clause}
            """
            count_result = await con.execute(count_query, params)
            total_count = (await count_result.fetchone())[0]

            # Get paginated results
            offset = (page - 1) * page_size
            params.extend([page_size, offset])

            # Conditionally include attributes field
            # When not including full attributes, return minimal subset for UI display
            if include_attributes:
                attributes_field = "et.attributes"
            else:
                attributes_field = """
                    jsonb_build_object(
                        'git_hash', et.attributes->>'git_hash',
                        'output_log_path', et.attributes->>'output_log_path',
                        'stderr_log_path', et.attributes->>'stderr_log_path',
                        'stdout_log_path', et.attributes->>'stdout_log_path',
                        'details', et.attributes->'details'
                    ) as attributes
                """.strip()

            async with con.cursor(row_factory=class_row(EvalTaskWithPolicyName)) as cur:
                await cur.execute(
                    f"""
                    SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                           et.assignee, et.created_at, {attributes_field}, et.retries,
                           p.name as policy_name, p.url as policy_url, et.user_id, et.updated_at
                    FROM eval_tasks et
                    LEFT JOIN policies p ON et.policy_id = p.id
                    WHERE {where_clause}
                    ORDER BY et.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    params,
                )
                tasks = await cur.fetchall()

            return tasks, total_count

    async def get_git_hashes_for_workers(self, assignees: list[str]) -> dict[str, list[str]]:
        async with self.connect() as con:
            if not assignees:
                return {}

            # Use ANY() for proper list handling in PostgreSQL
            queryRes = await con.execute(
                "SELECT DISTINCT assignee, attributes->>'git_hash' FROM eval_tasks WHERE assignee = ANY(%s)",
                (assignees,),
            )
            rows = await queryRes.fetchall()
            res: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                if row[1]:  # Only add non-null git hashes
                    res[row[0]].append(row[1])
            return res

    async def create_cogames_submission(
        self, submission_id: uuid.UUID, user_id: str, s3_path: str, name: str | None = None
    ) -> uuid.UUID:
        """Create a new CoGames policy submission with a specific ID."""
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO cogames_policy_submissions (id, user_id, name, s3_path)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (submission_id, user_id, name, s3_path),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to create CoGames submission")
            return row[0]
