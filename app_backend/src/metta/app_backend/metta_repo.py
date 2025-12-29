import json
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from psycopg import Connection
from psycopg.rows import class_row, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import BaseModel, Field, field_validator

from metta.app_backend.config import settings
from metta.app_backend.leaderboard_constants import (
    LEADERBOARD_CANDIDATE_COUNT_KEY,
    LEADERBOARD_LADYBUG_COUNT_KEY,
    LEADERBOARD_SCENARIO_KEY,
    LEADERBOARD_THINKY_COUNT_KEY,
    REPLACEMENT_BASELINE_MEAN,
)
from metta.app_backend.migrations import MIGRATIONS
from metta.app_backend.schema_manager import run_migrations
from metta.app_backend.value_over_replacement import (
    RunningStats,
    compute_overall_vor_from_stats,
)
from metta.common.util.memoization import memoize

TaskStatus = Literal["unprocessed", "running", "canceled", "done", "error", "system_error"]
FinishedTaskStatus = Literal["done", "error", "canceled", "system_error"]


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    clear_assignee: bool = False
    status_details: dict[str, Any] = Field(default_factory=dict)


class EvalTaskRow(BaseModel):
    """Row model that matches the eval_tasks table with latest attempt data."""

    model_config = {"from_attributes": True}

    id: int
    command: str
    data_uri: str | None
    git_hash: str | None
    attributes: dict[str, Any]
    user_id: str
    created_at: datetime
    is_finished: bool
    latest_attempt_id: int | None

    # Fields from the latest attempt (populated via JOIN)
    # Note: attempt_number will be 0 for new tasks, status will be 'unprocessed'
    attempt_number: int | None = 0
    status: TaskStatus = "unprocessed"
    status_details: dict[str, Any] | None = None
    assigned_at: datetime | None = None
    assignee: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    output_log_path: str | None = None


class TaskAttemptRow(BaseModel):
    """Row model for task_attempts table."""

    model_config = {"from_attributes": True}

    id: int
    task_id: int
    attempt_number: int
    assigned_at: datetime | None
    assignee: str | None
    started_at: datetime | None
    finished_at: datetime | None
    output_log_path: str | None
    status: TaskStatus
    status_details: dict[str, Any] | None


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
    created_at: datetime
    user_id: str
    attributes: dict[str, Any]
    version_count: int


class PolicyVersionRow(BaseModel):
    id: uuid.UUID
    internal_id: int
    policy_id: uuid.UUID
    version: int
    s3_path: str | None
    git_hash: str | None
    policy_spec: dict[str, Any]
    attributes: dict[str, Any]
    created_at: datetime


class PolicyVersionWithName(BaseModel):
    id: uuid.UUID
    internal_id: int
    policy_id: uuid.UUID
    version: int
    s3_path: str | None
    git_hash: str | None
    policy_spec: dict[str, Any]
    attributes: dict[str, Any]
    created_at: datetime
    name: str


class PublicPolicyVersionRow(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    created_at: datetime
    policy_created_at: datetime
    user_id: str
    name: str
    version: int
    tags: dict[str, str] = Field(default_factory=dict)
    version_count: int | None = None


class EpisodeReplay(BaseModel):
    episode_id: uuid.UUID
    replay_url: str


class EpisodeWithTags(BaseModel):
    id: uuid.UUID
    primary_pv_id: Optional[uuid.UUID]
    replay_url: Optional[str]
    thumbnail_url: Optional[str]
    attributes: dict[str, Any] = Field(default_factory=dict)
    eval_task_id: Optional[uuid.UUID]
    created_at: datetime
    tags: dict[str, str] = Field(default_factory=dict)
    avg_rewards: dict[uuid.UUID, float] = Field(default_factory=dict)
    policy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)

    # We need this because we don't insert a json object into attributes, we insert a string reflecting the json object.
    @field_validator("attributes", mode="before")
    @classmethod
    def _ensure_dict_attributes(cls, value: Any) -> dict[str, Any]:
        """Coerce JSON strings into dictionaries so validation doesn't fail."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("attributes must be a JSON object")
            return parsed
        raise ValueError("attributes must be a dictionary")


class LeaderboardPolicyEntry(BaseModel):
    policy_version: PublicPolicyVersionRow
    scores: dict[str, float]
    avg_score: float | None = None
    overall_vor: float | None = None  # Value Over Replacement (fetched separately)
    replays: dict[str, list[EpisodeReplay]] = Field(default_factory=dict)
    score_episode_ids: dict[str, uuid.UUID | None] = Field(default_factory=dict)


logger = logging.getLogger(name="metta_repo")


class MettaRepo:
    def __init__(self, db_uri: str) -> None:
        self.db_uri = db_uri
        self._pool: AsyncConnectionPool | None = None
        # Run migrations synchronously during initialization
        if settings.RUN_MIGRATIONS:
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

    async def create_eval_task(
        self,
        command: str,
        user_id: str,
        attributes: dict[str, Any],
        git_hash: str | None = None,
        data_uri: str | None = None,
    ) -> EvalTaskRow:
        async with self.connect() as con:
            # Insert the task
            result = await con.execute(
                """
                INSERT INTO eval_tasks (command, data_uri, git_hash, attributes, user_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (command, data_uri, git_hash, Jsonb(attributes), user_id),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to create eval task")
            task_id = row[0]

            # Create the first attempt
            result2 = await con.execute(
                """
                INSERT INTO task_attempts (task_id, attempt_number, status)
                VALUES (%s, 0, 'unprocessed')
                RETURNING id
                """,
                (task_id,),
            )
            row2 = await result2.fetchone()
            if row2 is None:
                raise RuntimeError("Failed to create first attempt")
            attempt_id = row2[0]

            # Update the task with the latest_attempt_id
            await con.execute(
                """
                UPDATE eval_tasks
                SET latest_attempt_id = %s
                WHERE id = %s
                """,
                (attempt_id, task_id),
            )

            # Fetch and return the complete task directly within the same transaction
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute("SELECT * FROM eval_tasks_view WHERE id = %s", (task_id,))
                task = await cur.fetchone()
                if task is None:
                    raise RuntimeError("Failed to retrieve created task")
                return task

    async def get_available_tasks(self, limit: int = 200) -> list[EvalTaskRow]:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute(
                    """
                    SELECT * FROM eval_tasks_view
                    WHERE status = 'unprocessed' AND assignee IS NULL AND is_finished = FALSE
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return await cur.fetchall()

    async def claim_tasks(
        self,
        task_ids: list[int],
        assignee: str,
    ) -> list[int]:
        if not task_ids:
            return []

        async with self.connect() as con:
            # Update the latest attempt for each task
            result = await con.execute(
                """
                UPDATE task_attempts
                SET assignee = %s, assigned_at = NOW()
                WHERE id IN (
                    SELECT latest_attempt_id FROM eval_tasks
                    WHERE id = ANY(%s) AND is_finished = FALSE
                )
                AND status = 'unprocessed'
                AND assignee IS NULL
                RETURNING task_id
                """,
                (assignee, task_ids),
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_claimed_tasks(self, assignee: str | None = None) -> list[EvalTaskRow]:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                if assignee is not None:
                    await cur.execute(
                        """
                        SELECT * FROM eval_tasks_view
                        WHERE assignee = %s AND is_finished = FALSE
                        ORDER BY created_at ASC
                        """,
                        (assignee,),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM eval_tasks_view
                        WHERE assignee IS NOT NULL AND is_finished = FALSE
                        ORDER BY created_at ASC
                        """
                    )
                return await cur.fetchall()

    async def start_task(self, task_id: int) -> None:
        async with self.connect() as con:
            await con.execute(
                """
                UPDATE task_attempts
                SET status = 'running', started_at = NOW()
                WHERE id = (SELECT latest_attempt_id FROM eval_tasks WHERE id = %s)
                """,
                (task_id,),
            )

    async def finish_task(
        self, task_id: int, status: FinishedTaskStatus, status_details: dict[str, Any], log_path: str | None = None
    ) -> None:
        async with self.connect() as con:
            # Update the current attempt
            await con.execute(
                """
                UPDATE task_attempts
                SET status = %s, finished_at = NOW(), status_details = %s, output_log_path = %s
                WHERE id = (SELECT latest_attempt_id FROM eval_tasks WHERE id = %s)
                """,
                (status, Jsonb(status_details), log_path, task_id),
            )

            # Get the current attempt number
            result = await con.execute(
                """
                SELECT attempt_number FROM task_attempts
                WHERE id = (SELECT latest_attempt_id FROM eval_tasks WHERE id = %s)
                """,
                (task_id,),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError(f"Failed to get attempt number for task {task_id}")
            current_attempt = row[0]

            # Check if we should mark the task as finished or create a new attempt
            should_finish = status != "system_error" or current_attempt >= 2  # 0, 1, 2 = 3 attempts

            if should_finish:
                # Mark the task as finished
                await con.execute(
                    """
                    UPDATE eval_tasks
                    SET is_finished = TRUE
                    WHERE id = %s
                    """,
                    (task_id,),
                )
            else:
                # Create a new attempt for retry
                await con.execute(
                    """
                    INSERT INTO task_attempts (task_id, attempt_number, status)
                    VALUES (%s, %s, 'unprocessed')
                    """,
                    (task_id, current_attempt + 1),
                )

                # Update latest_attempt_id
                await con.execute(
                    """
                    UPDATE eval_tasks
                    SET latest_attempt_id = (
                        SELECT id FROM task_attempts WHERE task_id = %s ORDER BY attempt_number DESC LIMIT 1
                    )
                    WHERE id = %s
                    """,
                    (task_id, task_id),
                )

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

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> EvalTaskRow | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute(
                    """
                    SELECT * FROM eval_tasks_view
                    WHERE assignee = %s AND assigned_at IS NOT NULL
                    ORDER BY assigned_at DESC
                    LIMIT 1
                    """,
                    (assignee,),
                )
                return await cur.fetchone()

    async def get_task_by_id(self, task_id: int) -> EvalTaskRow | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute("SELECT * FROM eval_tasks_view WHERE id = %s", (task_id,))
                return await cur.fetchone()

    async def get_task_attempts(self, task_id: int) -> list[TaskAttemptRow]:
        """Get all attempts for a task, ordered by attempt_number."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(TaskAttemptRow)) as cur:
                await cur.execute(
                    """
                    SELECT * FROM task_attempts
                    WHERE task_id = %s
                    ORDER BY attempt_number ASC
                    """,
                    (task_id,),
                )
                return await cur.fetchall()

    async def get_all_tasks(
        self,
        limit: int = 500,
        statuses: list[TaskStatus] | None = None,
        git_hash: str | None = None,
    ) -> list[EvalTaskRow]:
        async with self.connect() as con:
            # Build the WHERE clause dynamically
            where_conditions = []
            params = []

            if statuses:
                placeholders = ", ".join(["%s"] * len(statuses))
                where_conditions.append(f"status IN ({placeholders})")
                params.extend(statuses)

            if git_hash:
                where_conditions.append("git_hash = %s")
                params.append(git_hash)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            params.append(limit)

            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM eval_tasks_view
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                return await cur.fetchall()

    async def get_tasks_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        status: str | None = None,
        assignee: str | None = None,
        user_id: str | None = None,
        command: str | None = None,
        created_at: str | None = None,
        assigned_at: str | None = None,
    ) -> tuple[list[EvalTaskRow], int]:
        async with self.connect() as con:
            where_conditions = []
            params = []

            # Add text-based filters using ILIKE for case-insensitive substring search
            if status:
                # Use exact match for status since it's an enum-like field
                where_conditions.append("status = %s")
                params.append(status)

            if assignee:
                where_conditions.append("assignee ILIKE %s")
                params.append(f"%{assignee}%")

            if user_id:
                where_conditions.append("user_id ILIKE %s")
                params.append(f"%{user_id}%")

            if command:
                where_conditions.append("command ILIKE %s")
                params.append(f"%{command}%")

            if created_at:
                where_conditions.append("CAST(created_at AS TEXT) ILIKE %s")
                params.append(f"%{created_at}%")

            if assigned_at:
                where_conditions.append("CAST(assigned_at AS TEXT) ILIKE %s")
                params.append(f"%{assigned_at}%")

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # Get total count
            count_query = f"""
                SELECT COUNT(*)
                FROM eval_tasks_view
                WHERE {where_clause}
            """
            count_result = await con.execute(count_query, params)
            result_row = await count_result.fetchone()
            total_count: int = result_row[0] if result_row else 0

            # Get paginated results
            offset = (page - 1) * page_size
            params.extend([page_size, offset])

            async with con.cursor(row_factory=class_row(EvalTaskRow)) as cur:
                await cur.execute(
                    f"""
                    SELECT * FROM eval_tasks_view
                    WHERE {where_clause}
                    ORDER BY created_at DESC
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
                """
                SELECT DISTINCT assignee, git_hash
                FROM eval_tasks_view
                WHERE assignee = ANY(%s)
                """,
                (assignees,),
            )
            rows = await queryRes.fetchall()
            res: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                if row[1]:  # Only add non-null git hashes
                    res[row[0]].append(row[1])
            return res

    # Stats queries
    async def upsert_policy(self, name: str, user_id: str, attributes: dict[str, Any]) -> uuid.UUID:
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO policies (name, user_id, attributes)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, name) DO NOTHING
                RETURNING id
                """,
                (name, user_id, Jsonb(attributes)),
            )
            row = await result.fetchone()
            if row is not None:
                return row[0]
            else:
                # Policy already exists, get the id
                result = await con.execute(
                    """
                    SELECT id FROM policies WHERE user_id = %s AND name = %s
                    """,
                    (user_id, name),
                )
                row = await result.fetchone()
                if row is not None:
                    return row[0]
                else:
                    raise ValueError(f"Policy {name} not found")

    async def get_latest_policy_version(self, policy_id: uuid.UUID) -> int | None:
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT MAX(version) FROM policy_versions WHERE policy_id = %s
                """,
                (policy_id,),
            )
            res = await result.fetchone()
            if res:
                return res[0]

    async def create_policy_version(
        self,
        policy_id: uuid.UUID,
        s3_path: str | None,
        git_hash: str | None,
        policy_spec: dict[str, Any],
        attributes: dict[str, Any],
    ) -> uuid.UUID:
        async with self.connect() as con:
            latest_version = await self.get_latest_policy_version(policy_id)
            next_version = latest_version + 1 if latest_version is not None else 1

            result = await con.execute(
                """
                INSERT INTO policy_versions (policy_id, version, s3_path, git_hash, policy_spec, attributes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (policy_id, next_version, s3_path, git_hash, Jsonb(policy_spec), Jsonb(attributes)),
            )
            row = await result.fetchone()
            if row is None:
                raise ValueError("Failed to create policy version")
            return row[0]

    async def get_policy_version_with_name(self, policy_version_id: uuid.UUID) -> PolicyVersionWithName | None:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(PolicyVersionWithName)) as cur:
                await cur.execute(
                    """
                SELECT pv.*, p.name
                FROM policy_versions pv JOIN policies p ON pv.policy_id = p.id
                WHERE pv.id = %s""",
                    (policy_version_id,),
                )
                return await cur.fetchone()

    async def get_public_policy_version_by_id(self, policy_version_id: uuid.UUID) -> PublicPolicyVersionRow | None:
        """Get a single policy version with public fields by ID.

        Returns None if the policy version does not exist.
        """
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(PublicPolicyVersionRow)) as cur:
                await cur.execute(
                    """
                    SELECT
                        pv.id,
                        pv.policy_id,
                        pv.created_at,
                        p.created_at AS policy_created_at,
                        p.user_id,
                        p.name,
                        pv.version
                    FROM policy_versions pv
                    JOIN policies p ON pv.policy_id = p.id
                    WHERE pv.id = %s
                    """,
                    (policy_version_id,),
                )
                return await cur.fetchone()

    async def get_user_policy_versions(self, user_id: str) -> list[PublicPolicyVersionRow]:
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(PublicPolicyVersionRow)) as cur:
                await cur.execute(
                    """
                    SELECT
                        pv.id,
                        pv.policy_id,
                        pv.created_at,
                        p.created_at AS policy_created_at,
                        user_id,
                        p.name,
                        pv.version
                    FROM policy_versions AS pv, policies AS p
                    WHERE pv.policy_id = p.id AND p.user_id = %s
                    ORDER BY pv.created_at DESC, pv.version DESC
                    """,
                    (user_id,),
                )
                return await cur.fetchall()

    async def get_policies(
        self,
        name_exact: str | None = None,
        name_fuzzy: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PolicyRow], int]:
        async with self.connect() as con:
            where_conditions: list[str] = []
            params: list[Any] = []

            if name_exact:
                where_conditions.append("p.name = %s")
                params.append(name_exact)

            if name_fuzzy:
                where_conditions.append("p.name ILIKE %s")
                params.append(f"%{name_fuzzy}%")

            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

            count_query = f"SELECT COUNT(*) FROM policies p {where_clause}"
            count_result = await con.execute(count_query, params)  # type: ignore[arg-type]
            result_row = await count_result.fetchone()
            total_count: int = result_row[0] if result_row else 0

            params.extend([limit, offset])

            policy_query = f"""
                    SELECT
                        p.id,
                        p.name,
                        p.created_at,
                        p.user_id,
                        p.attributes,
                        COALESCE(vc.version_count, 0) AS version_count
                    FROM policies p
                    LEFT JOIN (
                        SELECT policy_id, COUNT(*) AS version_count
                        FROM policy_versions
                        GROUP BY policy_id
                    ) vc ON p.id = vc.policy_id
                    {where_clause}
                    ORDER BY p.created_at DESC
                    LIMIT %s OFFSET %s
                    """
            async with con.cursor(row_factory=class_row(PolicyRow)) as cur:
                await cur.execute(policy_query, params)  # type: ignore[arg-type]
                rows = await cur.fetchall()

            return rows, total_count

    async def get_policy_versions(
        self,
        name_exact: str | None = None,
        name_fuzzy: str | None = None,
        version: int | None = None,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PublicPolicyVersionRow], int]:
        async with self.connect() as con:
            where_conditions: list[str] = []
            params: list[Any] = []

            if policy_version_ids:
                where_conditions.append("pv.id = ANY(%s)")
                params.append(policy_version_ids)

            if name_exact:
                where_conditions.append("p.name = %s")
                params.append(name_exact)

            if name_fuzzy:
                where_conditions.append("p.name ILIKE %s")
                params.append(f"%{name_fuzzy}%")

            if version is not None:
                where_conditions.append("pv.version = %s")
                params.append(version)

            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

            count_query = f"""
                SELECT COUNT(DISTINCT pv.policy_id)
                FROM policy_versions pv
                JOIN policies p ON pv.policy_id = p.id
                {where_clause}
            """
            count_result = await con.execute(count_query, params)  # type: ignore[arg-type]
            result_row = await count_result.fetchone()
            total_count: int = result_row[0] if result_row else 0

            params.extend([limit, offset])

            version_query = f"""
                    SELECT
                        pv.id,
                        pv.policy_id,
                        pv.created_at,
                        p.created_at AS policy_created_at,
                        p.user_id,
                        p.name,
                        pv.version,
                        pv.version AS version_count
                    FROM policy_versions pv
                    JOIN policies p ON pv.policy_id = p.id
                    {where_clause}
                    ORDER BY pv.created_at DESC
                    LIMIT %s OFFSET %s
                    """
            async with con.cursor(row_factory=class_row(PublicPolicyVersionRow)) as cur:
                await cur.execute(version_query, params)  # type: ignore[arg-type]
                rows = await cur.fetchall()

            return rows, total_count

    async def get_versions_for_policy(
        self,
        policy_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[PublicPolicyVersionRow], int]:
        async with self.connect() as con:
            count_query = """
                SELECT COUNT(*)
                FROM policy_versions pv
                WHERE pv.policy_id = %s
            """
            count_result = await con.execute(count_query, (policy_id,))
            result_row = await count_result.fetchone()
            total_count: int = result_row[0] if result_row else 0

            async with con.cursor(row_factory=class_row(PublicPolicyVersionRow)) as cur:
                await cur.execute(
                    """
                    SELECT
                        pv.id,
                        pv.policy_id,
                        pv.created_at,
                        p.created_at AS policy_created_at,
                        p.user_id,
                        p.name,
                        pv.version
                    FROM policy_versions pv
                    JOIN policies p ON pv.policy_id = p.id
                    WHERE pv.policy_id = %s
                    ORDER BY pv.version DESC
                    LIMIT %s OFFSET %s
                    """,
                    (policy_id, limit, offset),
                )
                rows = await cur.fetchall()

            return rows, total_count

    async def upsert_policy_version_tags(self, policy_version_id: uuid.UUID, tags: dict[str, str]) -> None:
        if not tags:
            return

        rows = [(policy_version_id, key, value) for key, value in tags.items()]

        async with self.connect() as con:
            async with con.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO policy_version_tags (policy_version_id, key, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (policy_version_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    rows,
                )

    async def record_episode(
        self,
        id: UUID,
        data_uri: str,
        primary_pv_id: uuid.UUID | None,
        replay_url: str | None,
        attributes: dict[str, Any],
        eval_task_id: uuid.UUID | None,
        thumbnail_url: str | None,
        tags: list[tuple[str, str]],
        policy_versions: list[tuple[uuid.UUID, int]],  # pv_id, num_agents
        policy_metrics: list[tuple[uuid.UUID, str, float]],  # pv_id, metric_name, metric_value
    ) -> uuid.UUID:
        async with self.connect() as con:
            # Insert into episodes table
            result = await con.execute(
                """
                INSERT INTO episodes (
                    id,
                    data_uri,
                    primary_pv_id,
                    replay_url,
                    thumbnail_url,
                    attributes,
                    eval_task_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                ) RETURNING internal_id
                """,
                (id, data_uri, primary_pv_id, replay_url, thumbnail_url, Jsonb(attributes), eval_task_id),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert episode record")
            episode_internal_id = row[0]

            # Insert episode policies in bulk
            async with con.cursor() as cur:
                rows = [(id, pv_id, num_agents) for pv_id, num_agents in policy_versions]
                await cur.executemany(
                    """
                    INSERT INTO episode_policies (episode_id, policy_version_id, num_agents)
                    VALUES (%s, %s, %s)
                    """,
                    rows,
                )

            # Get internal_id for each policy version UUID
            pv_uuid_to_internal_id: dict[uuid.UUID, int] = {}
            if policy_metrics:
                pv_uuids = list({pv_id for pv_id, _, _ in policy_metrics})
                result = await con.execute(
                    """
                    SELECT id, internal_id FROM policy_versions WHERE id = ANY(%s)
                    """,
                    (pv_uuids,),
                )
                rows = await result.fetchall()
                for row in rows:
                    pv_uuid_to_internal_id[row[0]] = row[1]

            # Insert episode policy metrics in bulk
            async with con.cursor() as cur:
                rows = [
                    (episode_internal_id, pv_uuid_to_internal_id[pv_id], metric_name, metric_value)
                    for pv_id, metric_name, metric_value in policy_metrics
                ]
                await cur.executemany(
                    """
                    INSERT INTO episode_policy_metrics (episode_internal_id, pv_internal_id, metric_name, value)
                    VALUES (%s, %s, %s, %s)
                    """,
                    rows,
                )

            # Insert episode tags in bulk
            async with con.cursor() as cur:
                rows = [(id, key, value) for key, value in tags]
                await cur.executemany(
                    """
                    INSERT INTO episode_tags (episode_id, key, value)
                    VALUES (%s, %s, %s)
                    """,
                    rows,
                )

            return id

    async def get_leaderboard_policies(
        self,
        policy_version_tags: dict[str, str],
        score_group_episode_tag: str,
        user_id: str | None = None,
        policy_version_id: uuid.UUID | None = None,
    ) -> list[LeaderboardPolicyEntry]:
        """Return leaderboard entries for policy versions matching the given tag filters."""
        policy_conditions: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            policy_conditions.append("pol.user_id = %s")
            params.append(user_id)

        if policy_version_id is not None:
            policy_conditions.append("pv.id = %s")
            params.append(policy_version_id)

        for idx, (tag_key, tag_value) in enumerate(policy_version_tags.items()):
            policy_conditions.append(
                f"""EXISTS (
                        SELECT 1 FROM policy_version_tags pvt_{idx}
                        WHERE pvt_{idx}.policy_version_id = pv.id
                          AND pvt_{idx}.key = %s
                          AND pvt_{idx}.value = %s
                    )"""
            )
            params.extend([tag_key, tag_value])

        where_clause = f"WHERE {' AND '.join(policy_conditions)}" if policy_conditions else ""

        policy_query = f"""
SELECT
    pv.id,
    pv.policy_id,
    pv.created_at,
    pol.created_at AS policy_created_at,
    pol.user_id,
    pol.name,
    pv.version,
    '{{}}'::jsonb AS tags
FROM policy_versions pv
JOIN policies pol ON pol.id = pv.policy_id
{where_clause}
ORDER BY pol.created_at DESC, pv.created_at DESC
"""

        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(PublicPolicyVersionRow)) as cur:
                await cur.execute(policy_query, params)  # type: ignore
                policy_rows = await cur.fetchall()

            if not policy_rows:
                return []

            policy_version_ids = [row.id for row in policy_rows]
            scores_by_policy: dict[uuid.UUID, dict[str, float]] = {pv_id: {} for pv_id in policy_version_ids}
            score_episode_ids: dict[uuid.UUID, dict[str, uuid.UUID | None]] = {
                pv_id: {} for pv_id in policy_version_ids
            }

            async with con.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
SELECT policy_version_id, key, value
FROM policy_version_tags
WHERE policy_version_id = ANY(%s)
""",
                    (policy_version_ids,),
                )
                tag_rows = await cur.fetchall()

            tags_by_policy: dict[uuid.UUID, dict[str, str]] = {pv_id: {} for pv_id in policy_version_ids}
            for tag_row in tag_rows:
                tags_by_policy[tag_row["policy_version_id"]][tag_row["key"]] = tag_row["value"]

            if score_group_episode_tag:
                scores_query = """
SELECT
    pv.id AS policy_version_id,
    et.key AS tag_key,
    et.value AS tag_value,
    AVG(epm.value / ep.num_agents) AS avg_reward_per_agent,
    (ARRAY_AGG(e.id ORDER BY e.created_at DESC, e.id DESC))[1] AS latest_episode_id
FROM policy_versions pv
JOIN episode_policies ep ON ep.policy_version_id = pv.id
JOIN episodes e ON e.id = ep.episode_id
JOIN episode_policy_metrics epm
    ON epm.episode_internal_id = e.internal_id
   AND epm.pv_internal_id = pv.internal_id
JOIN episode_tags et ON et.episode_id = e.id
WHERE epm.metric_name = 'reward'
  AND et.key = %s
  AND pv.id = ANY(%s)
GROUP BY pv.id, et.key, et.value
"""

                async with con.cursor(row_factory=dict_row) as cur:
                    await cur.execute(scores_query, (score_group_episode_tag, policy_version_ids))
                    score_rows = await cur.fetchall()

                for score_row in score_rows:
                    pv_id = score_row["policy_version_id"]
                    tag_identifier = f"{score_row['tag_key']}:{score_row['tag_value']}"
                    scores_by_policy.setdefault(pv_id, {})[tag_identifier] = float(score_row["avg_reward_per_agent"])
                    score_episode_ids.setdefault(pv_id, {})[tag_identifier] = score_row["latest_episode_id"]

        entries: list[LeaderboardPolicyEntry] = []
        for policy_row in policy_rows:
            pv_id = policy_row.id
            scores = scores_by_policy.get(pv_id, {})
            avg_score = sum(scores.values()) / len(scores) if scores else None
            policy_version = policy_row.model_copy(update={"tags": tags_by_policy.get(pv_id, {})})
            entries.append(
                LeaderboardPolicyEntry(
                    policy_version=policy_version,
                    scores=dict(scores),
                    avg_score=avg_score,
                    score_episode_ids=dict(score_episode_ids.get(pv_id, {})),
                )
            )

        entries.sort(
            key=lambda entry: (
                0 if entry.avg_score is not None else 1,
                -(entry.avg_score or 0.0),
                -entry.policy_version.created_at.timestamp(),
            )
        )
        return entries

    @memoize(max_age=60.0)
    async def _get_vor_stats(
        self, policy_version_ids: tuple[uuid.UUID | str, ...]
    ) -> defaultdict[uuid.UUID, defaultdict[int, RunningStats]]:
        query = f"""
        SELECT
            ep.policy_version_id,
            et_cand.value::int AS candidate_count,
            et_thinky.value::int AS thinky_count,
            et_lady.value::int AS ladybug_count,
            epm.value / NULLIF(ep.num_agents, 0) AS avg_reward
        FROM episodes e
        JOIN episode_tags et_scen
            ON et_scen.episode_id = e.id AND et_scen.key = '{LEADERBOARD_SCENARIO_KEY}'
        JOIN episode_tags et_cand
            ON et_cand.episode_id = e.id AND et_cand.key = '{LEADERBOARD_CANDIDATE_COUNT_KEY}'
        LEFT JOIN episode_tags et_thinky
            ON et_thinky.episode_id = e.id AND et_thinky.key = '{LEADERBOARD_THINKY_COUNT_KEY}'
        LEFT JOIN episode_tags et_lady
            ON et_lady.episode_id = e.id AND et_lady.key = '{LEADERBOARD_LADYBUG_COUNT_KEY}'
        JOIN episode_policies ep ON ep.episode_id = e.id
        JOIN policy_versions pv ON pv.id = ep.policy_version_id
        JOIN episode_policy_metrics epm
            ON epm.episode_internal_id = e.internal_id AND epm.pv_internal_id = pv.internal_id
        WHERE e.primary_pv_id = ANY(%s)
            AND epm.metric_name = 'reward'
            AND ep.num_agents > 0
        """
        async with self.connect() as con:
            async with con.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (list(policy_version_ids),))
                rows = await cur.fetchall()
        stats_by_policy: defaultdict[uuid.UUID, defaultdict[int, RunningStats]] = defaultdict(
            lambda: defaultdict(RunningStats)
        )
        for row in rows:
            candidate_count = row.get("candidate_count")
            avg_reward = row.get("avg_reward")
            if candidate_count is None or avg_reward is None:
                continue
            candidate_count = int(candidate_count)
            reward = float(avg_reward)

            if candidate_count == 0:
                # Replacement: weight = thinky + ladybug
                thinky_count = int(row.get("thinky_count") or 0)
                ladybug_count = int(row.get("ladybug_count") or 0)
                weight = thinky_count + ladybug_count
            else:
                weight = candidate_count

            stats_by_policy[row["policy_version_id"]][candidate_count].update(reward, weight=weight)
        return stats_by_policy

    async def get_leaderboard_policies_with_vor(
        self,
        policy_version_tags: dict[str, str],
        score_group_episode_tag: str,
    ) -> list[LeaderboardPolicyEntry]:
        """Return leaderboard entries with overall_vor computed for each policy."""
        entries = await self.get_leaderboard_policies(
            policy_version_tags=policy_version_tags,
            score_group_episode_tag=score_group_episode_tag,
            user_id=None,
            policy_version_id=None,
        )
        candidate_vor_stats = await self._get_vor_stats(tuple(entry.policy_version.id for entry in entries))

        for entry in entries:
            pv_id = entry.policy_version.id
            candidate_stats = candidate_vor_stats.get(pv_id, {})
            if candidate_stats:
                entry.overall_vor = compute_overall_vor_from_stats(candidate_stats, REPLACEMENT_BASELINE_MEAN)

        return entries

    async def get_episodes(
        self,
        *,
        primary_policy_version_ids: Optional[list[uuid.UUID]] = None,
        episode_ids: Optional[list[uuid.UUID]] = None,
        tag_filters: Optional[dict[str, Optional[list[str]]]] = None,
        limit: Optional[int] = 200,
        offset: int = 0,
    ) -> list[EpisodeWithTags]:
        """Fetch episodes with optional filters and tag aggregation."""
        where_conditions: list[str] = []
        params: list[Any] = []

        if primary_policy_version_ids:
            where_conditions.append("e.primary_pv_id = ANY(%s)")
            params.append(primary_policy_version_ids)

        if episode_ids:
            where_conditions.append("e.id = ANY(%s)")
            params.append(episode_ids)

        if tag_filters:
            for idx, (tag_key, tag_values) in enumerate(tag_filters.items()):
                if tag_values:
                    where_conditions.append(
                        f"""EXISTS (
                            SELECT 1 FROM episode_tags et_{idx}
                            WHERE et_{idx}.episode_id = e.id
                              AND et_{idx}.key = %s
                              AND et_{idx}.value = ANY(%s)
                        )"""
                    )
                    params.extend([tag_key, tag_values])
                else:
                    where_conditions.append(
                        f"""EXISTS (
                            SELECT 1 FROM episode_tags et_{idx}
                            WHERE et_{idx}.episode_id = e.id
                              AND et_{idx}.key = %s
                        )"""
                    )
                    params.append(tag_key)

        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(limit)
        if offset > 0:
            limit_clause += " OFFSET %s" if limit_clause else "OFFSET %s"
            params.append(offset)

        query = f"""
WITH episode_tags_agg AS (
    SELECT episode_id, jsonb_object_agg(key, value) AS tags
    FROM episode_tags
    GROUP BY episode_id
),
episode_policy_metrics AS (
    SELECT
        e_sub.id AS episode_id,
        pv.id AS policy_version_id,
        epm.metric_name,
        CASE
            WHEN epm.metric_name = 'reward' AND ep.num_agents > 0 THEN epm.value / ep.num_agents
            ELSE epm.value
        END AS metric_value
    FROM episodes e_sub
    JOIN episode_policies ep ON ep.episode_id = e_sub.id
    JOIN policy_versions pv ON pv.id = ep.policy_version_id
    JOIN episode_policy_metrics epm
        ON epm.episode_internal_id = e_sub.internal_id
       AND epm.pv_internal_id = pv.internal_id
),
episode_policy_metrics_map AS (
    SELECT
        episode_id,
        jsonb_object_agg(
            policy_version_id::text,
            metrics_by_policy
        ) AS policy_metrics
    FROM (
        SELECT
            episode_id,
            policy_version_id,
            jsonb_object_agg(metric_name, metric_value) AS metrics_by_policy
        FROM episode_policy_metrics
        GROUP BY episode_id, policy_version_id
    ) grouped
    GROUP BY episode_id
),
episode_avg_rewards AS (
    SELECT
        episode_id,
        jsonb_object_agg(policy_version_id::text, metric_value) AS avg_rewards
    FROM episode_policy_metrics
    WHERE metric_name = 'reward'
    GROUP BY episode_id
)
SELECT
    e.id,
    e.primary_pv_id,
    e.replay_url,
    e.thumbnail_url,
    COALESCE(e.attributes, '{{}}'::jsonb) AS attributes,
    e.eval_task_id,
    e.created_at,
    COALESCE(t.tags, '{{}}'::jsonb) AS tags,
    COALESCE(r.avg_rewards, '{{}}'::jsonb) AS avg_rewards,
    COALESCE(pm.policy_metrics, '{{}}'::jsonb) AS policy_metrics
FROM episodes e
LEFT JOIN episode_tags_agg t ON t.episode_id = e.id
LEFT JOIN episode_avg_rewards r ON r.episode_id = e.id
LEFT JOIN episode_policy_metrics_map pm ON pm.episode_id = e.id
{where_clause}
ORDER BY e.created_at DESC
{limit_clause}
"""

        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(EpisodeWithTags)) as cur:
                await cur.execute(query, params)  # type: ignore
                rows = await cur.fetchall()

        for row in rows:
            # `class_row` returns a dict for this attr but doesn't coerce its inner types
            row.avg_rewards = {uuid.UUID(str(key)): value for key, value in row.avg_rewards.items()}
        return list(rows)
