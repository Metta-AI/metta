import hashlib
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Literal

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from metta.app_backend.query_logger import execute_single_row_query_and_log
from metta.app_backend.schema_manager import SqlMigration, run_migrations

TaskStatus = Literal["unprocessed", "canceled", "done", "error"]


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    clear_assignee: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


# This is a list of migrations that will be applied to the eval database.
# Do not change existing migrations, only add new ones.
MIGRATIONS = [
    SqlMigration(
        version=0,
        description="Initial eval schema",
        sql_statements=[
            """CREATE EXTENSION IF NOT EXISTS "uuid-ossp" """,
            """CREATE TABLE training_runs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                finished_at TIMESTAMP,
                status TEXT NOT NULL,
                url TEXT,
                attributes JSONB
            )""",
            """CREATE TABLE epochs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                run_id UUID NOT NULL,
                start_training_epoch INTEGER NOT NULL,
                end_training_epoch INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                attributes JSONB
            )""",
            """CREATE TABLE policies (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL,
                description TEXT,
                url TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                epoch_id UUID REFERENCES epochs(id),
                UNIQUE (name)
            )""",
            # This is slightly denormalized, in the sense that it is storing both the attributes of the env and
            # the attributes of the episode. We could imagine having separate (env, env_attributes) tables and having
            # a foreign key into envs in episodes.  However, I can imagine a lot of envs that are only used in a single
            # episode, so I'm not sure it's worth the extra complexity.
            """CREATE TABLE episodes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                primary_policy_id UUID NOT NULL REFERENCES policies(id),
                stats_epoch UUID REFERENCES epochs(id),
                replay_url TEXT,
                eval_name TEXT,
                simulation_suite TEXT,
                attributes JSONB
            )""",
            """CREATE TABLE episode_agent_policies (
                episode_id UUID NOT NULL REFERENCES episodes(id),
                policy_id UUID NOT NULL REFERENCES policies(id),
                agent_id INTEGER NOT NULL,
                PRIMARY KEY (episode_id, policy_id, agent_id)
            )""",
            """CREATE TABLE episode_agent_metrics (
                episode_id UUID NOT NULL REFERENCES episodes(id),
                agent_id INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL,
                PRIMARY KEY (episode_id, agent_id, metric)
            )""",
        ],
    ),
    SqlMigration(
        version=1,
        description="Add machine tokens table",
        sql_statements=[
            """CREATE TABLE machine_tokens (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expiration_time TIMESTAMP NOT NULL,
                last_used_at TIMESTAMP,
                UNIQUE (user_id, name)
            )""",
            """CREATE INDEX idx_machine_tokens_user_id ON machine_tokens(user_id)""",
            """CREATE INDEX idx_machine_tokens_token_hash ON machine_tokens(token_hash)""",
        ],
    ),
    SqlMigration(
        version=2,
        description="Make training run names unique",
        sql_statements=[
            """ALTER TABLE training_runs ADD CONSTRAINT training_runs_name_unique UNIQUE (user_id, name)""",
        ],
    ),
    SqlMigration(
        version=3,
        description="Remove machine token name uniqueness constraint",
        sql_statements=[
            """ALTER TABLE machine_tokens DROP CONSTRAINT machine_tokens_user_id_name_key""",
        ],
    ),
    SqlMigration(
        version=4,
        description="Add saved dashboards table",
        sql_statements=[
            """CREATE TABLE saved_dashboards (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                dashboard_state JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_saved_dashboards_user_id ON saved_dashboards(user_id)""",
            """CREATE VIEW episode_view AS
                SELECT id,
                  split_part(eval_name, '/', 1) as simulation_suite,
                  split_part(eval_name, '/', 2) as eval_name,
                  replay_url,
                  primary_policy_id,
                  stats_epoch,
                  attributes
                FROM episodes
                WHERE split_part(eval_name, '/', 1) IS NOT NULL AND split_part(eval_name, '/', 2) IS NOT NULL
            """,
        ],
    ),
    SqlMigration(
        version=5,
        description="Parse out eval category from eval name",
        sql_statements=[
            """ALTER TABLE episodes ADD COLUMN eval_category TEXT, ADD COLUMN env_name TEXT""",
            """UPDATE episodes SET eval_category = split_part(eval_name, '/', 1), """
            """env_name = split_part(eval_name, '/', 2)""",
            """CREATE INDEX idx_episodes_eval_category ON episodes(eval_category)""",
            """DROP VIEW episode_view""",
        ],
    ),
    SqlMigration(
        version=6,
        description="Add heatmap performance indexes",
        sql_statements=[
            # Critical index for episode_agent_metrics main query
            """ALTER TABLE episode_agent_metrics DROP CONSTRAINT episode_agent_metrics_pkey""",
            """CREATE INDEX idx_episode_agent_metrics_metric_episode_value
                ON episode_agent_metrics(metric, episode_id)
                INCLUDE (value)""",
            # Composite index for episodes eval filtering and joins
            """CREATE INDEX idx_episodes_eval_category_env_policy
               ON episodes(eval_category, env_name, primary_policy_id)""",
            # Index to optimize the JSON agent_groups lookup
            """CREATE INDEX idx_episodes_attributes_agent_groups
               ON episodes USING GIN ((attributes->'agent_groups'))""",
            # Index for policy-epoch joins
            """CREATE INDEX idx_policies_epoch_id ON policies(epoch_id)""",
            # Index for epochs run lookups
            """CREATE INDEX idx_epochs_run_id ON epochs(run_id, end_training_epoch DESC)""",
        ],
    ),
    SqlMigration(
        version=7,
        description="Add description field to training_runs table",
        sql_statements=[
            """ALTER TABLE training_runs ADD COLUMN description TEXT""",
        ],
    ),
    SqlMigration(
        version=8,
        description="Add tags field to training_runs table",
        sql_statements=[
            """ALTER TABLE training_runs ADD COLUMN tags TEXT[]""",
        ],
    ),
    SqlMigration(
        version=9,
        description="Add internal_id field to episodes table and episode_agent_metrics table",
        sql_statements=[
            """ALTER TABLE episodes ADD COLUMN internal_id SERIAL UNIQUE""",
            """ALTER TABLE episode_agent_metrics ADD COLUMN episode_internal_id INTEGER""",
        ],
    ),
    SqlMigration(
        version=10,
        description="Drop episode_agent_metrics.episode_id column",
        sql_statements=[
            """ALTER TABLE episode_agent_metrics ALTER COLUMN episode_internal_id SET NOT NULL""",
            """CREATE INDEX IF NOT EXISTS idx_episode_agent_metrics_metric_eiid_value
                ON episode_agent_metrics(metric, episode_internal_id)
                INCLUDE (value)""",
            """CREATE INDEX IF NOT EXISTS idx_episode_agent_metrics_episode_internal_id
                ON episode_agent_metrics(episode_internal_id)""",
            """DROP INDEX idx_episode_agent_metrics_metric_episode_value""",
            """ALTER TABLE episode_agent_metrics DROP COLUMN episode_id""",
        ],
    ),
    SqlMigration(
        version=11,
        description="Add eval_tasks table",
        sql_statements=[
            """CREATE TABLE eval_tasks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                policy_id UUID NOT NULL REFERENCES policies(id),
                sim_suite TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unprocessed',
                assigned_at TIMESTAMP,
                assignee TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                attributes JSONB
            )""",
            """CREATE INDEX idx_eval_tasks_status_assigned ON eval_tasks(status, assigned_at)""",
            """CREATE INDEX idx_eval_tasks_assignee ON eval_tasks(assignee)""",
            """CREATE INDEX idx_eval_tasks_policy_id ON eval_tasks(policy_id)""",
            """CREATE INDEX idx_eval_tasks_sim_suite ON eval_tasks(sim_suite)""",
            """CREATE INDEX idx_eval_tasks_unprocessed_assigned
               ON eval_tasks(assigned_at)
               WHERE status = 'unprocessed'""",
        ],
    ),
    SqlMigration(
        version=12,
        description="Add eval_task_id to episodes table",
        sql_statements=[
            """ALTER TABLE episodes ADD COLUMN eval_task_id UUID REFERENCES eval_tasks(id)""",
            """CREATE INDEX idx_episodes_eval_task_id ON episodes(eval_task_id)""",
        ],
    ),
    SqlMigration(
        version=13,
        description="Add episode_tags table",
        sql_statements=[
            """CREATE TABLE episode_tags (
                episode_id UUID NOT NULL REFERENCES episodes(id),
                tag TEXT NOT NULL,
                PRIMARY KEY (episode_id, tag)
            )""",
            """CREATE INDEX idx_episode_tags_episode_id ON episode_tags(episode_id)""",
            """CREATE INDEX idx_episode_tags_tag ON episode_tags(tag)""",
        ],
    ),
    SqlMigration(
        version=14,
        description="Add wide_episodes view for simplified episode filtering",
        sql_statements=[
            """CREATE VIEW wide_episodes AS
            SELECT
                e.id,
                e.created_at,
                e.primary_policy_id,
                e.stats_epoch,
                e.replay_url,
                e.eval_name,
                e.simulation_suite,
                e.eval_category,
                e.env_name,
                e.attributes,
                e.eval_task_id,
                p.name as policy_name,
                p.description as policy_description,
                p.url as policy_url,
                ep.start_training_epoch as epoch_start_training_epoch,
                ep.end_training_epoch as epoch_end_training_epoch,
                tr.id as training_run_id,
                tr.name as training_run_name,
                tr.user_id as training_run_user_id,
                tr.status as training_run_status,
                tr.url as training_run_url,
                tr.description as training_run_description,
                tr.tags as training_run_tags
            FROM episodes e
            LEFT JOIN policies p ON e.primary_policy_id = p.id
            LEFT JOIN epochs ep ON p.epoch_id = ep.id
            LEFT JOIN training_runs tr ON ep.run_id = tr.id
            """,
        ],
    ),
    SqlMigration(
        version=15,
        description="Add sweeps table for sweep coordination",
        sql_statements=[
            """CREATE TABLE sweeps (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL UNIQUE,
                project TEXT NOT NULL,
                entity TEXT NOT NULL,
                wandb_sweep_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'running',
                run_counter INTEGER NOT NULL DEFAULT 0,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_sweeps_name ON sweeps(name)""",
        ],
    ),
    SqlMigration(
        version=16,
        description="Add index on eval_tasks git_hash, assigned_at. And another on assigned_at",
        sql_statements=[
            """CREATE INDEX idx_eval_tasks_git_hash_assigned ON eval_tasks((attributes ->> 'git_hash'), assigned_at)""",
            """CREATE INDEX idx_eval_tasks_assigned_at ON eval_tasks(assigned_at)""",
        ],
    ),
    SqlMigration(
        version=17,
        description="Add retries column to eval_tasks table",
        sql_statements=[
            """ALTER TABLE eval_tasks ADD COLUMN retries INTEGER NOT NULL DEFAULT 0""",
            """CREATE INDEX idx_eval_tasks_retries ON eval_tasks(retries)""",
        ],
    ),
    SqlMigration(
        version=18,
        description="Add index on assignee, assigned_at, status",
        sql_statements=[
            """CREATE INDEX idx_eval_tasks_assignee_assigned_at_status
               ON eval_tasks(assignee, assigned_at, status)""",
        ],
    ),
    SqlMigration(
        version=19,
        description="Add internal_id to wide_episodes view",
        sql_statements=[
            """DROP VIEW wide_episodes""",
            """CREATE VIEW wide_episodes AS
            SELECT
                e.id,
                e.internal_id,
                e.created_at,
                e.primary_policy_id,
                e.stats_epoch,
                e.replay_url,
                e.eval_name,
                e.simulation_suite,
                e.eval_category,
                e.env_name,
                e.attributes,
                e.eval_task_id,
                p.name as policy_name,
                p.description as policy_description,
                p.url as policy_url,
                ep.start_training_epoch as epoch_start_training_epoch,
                ep.end_training_epoch as epoch_end_training_epoch,
                tr.id as training_run_id,
                tr.name as training_run_name,
                tr.user_id as training_run_user_id,
                tr.status as training_run_status,
                tr.url as training_run_url,
                tr.description as training_run_description,
                tr.tags as training_run_tags
            FROM episodes e
            LEFT JOIN policies p ON e.primary_policy_id = p.id
            LEFT JOIN epochs ep ON p.epoch_id = ep.id
            LEFT JOIN training_runs tr ON ep.run_id = tr.id
            """,
        ],
    ),
]


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

    async def get_policy_by_id(self, policy_id: uuid.UUID) -> dict[str, Any] | None:
        async with self.connect() as con:
            res = await con.execute(
                """
                SELECT id, name
                FROM policies
                WHERE id = %s
                """,
                (policy_id,),
            )
            row = await res.fetchone()
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "name": row[1],
            }

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

    async def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, str],
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
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None,
        eval_name: str | None,
        simulation_suite: str | None,
        replay_url: str | None,
        attributes: dict[str, Any],
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
    ) -> uuid.UUID:
        async with self.connect() as con:
            # Parse eval_category and env_name from eval_name
            eval_category = eval_name.split("/", 1)[0] if eval_name else None
            env_name = eval_name.split("/", 1)[1] if eval_name and "/" in eval_name else None

            # Insert into episodes table
            result = await con.execute(
                """
                INSERT INTO episodes (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    eval_category,
                    env_name,
                    primary_policy_id,
                    stats_epoch,
                    attributes,
                    eval_task_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id, internal_id
                """,
                (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    eval_category,
                    env_name,
                    primary_policy_id,
                    stats_epoch,
                    Jsonb(attributes),
                    eval_task_id,
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

    async def get_suites(self) -> list[str]:
        async with self.connect() as con:
            result = await con.execute("""
                SELECT DISTINCT eval_category
                FROM episodes
                WHERE eval_category IS NOT NULL AND env_name IS NOT NULL
                ORDER BY eval_category
            """)
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_metrics(self, suite: str) -> list[str]:
        """Get all available metrics for a given suite."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT DISTINCT eam.metric
                FROM episodes e
                JOIN episode_agent_metrics eam ON e.internal_id = eam.episode_internal_id
                WHERE e.eval_category = %s
                ORDER BY eam.metric
            """,
                (suite,),
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_group_ids(self, suite: str) -> list[str]:
        """Get all available group IDs for a given suite."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT DISTINCT jsonb_object_keys(e.attributes->'agent_groups') as group_id
                FROM episodes e
                WHERE e.eval_category = %s
                ORDER BY group_id
            """,
                (suite,),
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_training_runs(self) -> list[dict[str, Any]]:
        """Get all training runs."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, created_at, user_id, finished_at, status, url, description, tags
                FROM training_runs
                ORDER BY created_at DESC
                """
            )
            rows = await result.fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "created_at": row[2].isoformat(),
                    "user_id": row[3],
                    "finished_at": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "url": row[6],
                    "description": row[7],
                    "tags": row[8] or [],
                }
                for row in rows
            ]

    async def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific training run by ID."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return None

        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, created_at, user_id, finished_at, status, url, description, tags
                FROM training_runs
                WHERE id = %s
                """,
                (run_uuid,),
            )
            row = await result.fetchone()

            if row is None:
                return None

            return {
                "id": str(row[0]),
                "name": row[1],
                "created_at": row[2].isoformat(),
                "user_id": row[3],
                "finished_at": row[4].isoformat() if row[4] else None,
                "status": row[5],
                "url": row[6],
                "description": row[7],
                "tags": row[8] or [],
            }

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

    async def list_machine_tokens(self, user_id: str) -> list[dict[str, Any]]:
        """List all machine tokens for a user."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, created_at, expiration_time, last_used_at
                FROM machine_tokens
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = await result.fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "created_at": row[2],
                    "expiration_time": row[3],
                    "last_used_at": row[4],
                }
                for row in rows
            ]

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

    async def create_saved_dashboard(
        self,
        user_id: str,
        name: str,
        description: str | None,
        dashboard_type: str,
        dashboard_state: dict[str, Any],
    ) -> uuid.UUID:
        """Create a new saved dashboard (no upsert, always insert)."""
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO saved_dashboards (
                    user_id, name, description, type, dashboard_state
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, name, description, dashboard_type, Jsonb(dashboard_state)),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to create saved dashboard")
            return row[0]

    async def list_saved_dashboards(self) -> list[dict[str, Any]]:
        """List all saved dashboards."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, description, type, dashboard_state, created_at, updated_at, user_id
                FROM saved_dashboards
                ORDER BY updated_at DESC
                """
            )
            rows = await result.fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "description": row[2],
                    "type": row[3],
                    "dashboard_state": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "user_id": row[7],
                }
                for row in rows
            ]

    async def get_saved_dashboard(self, dashboard_id: str) -> dict[str, Any] | None:
        """Get a specific saved dashboard by ID."""
        try:
            dashboard_uuid = uuid.UUID(dashboard_id)
        except ValueError:
            return None

        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, description, type, dashboard_state, created_at, updated_at, user_id
                FROM saved_dashboards
                WHERE id = %s
                """,
                (dashboard_uuid,),
            )
            row = await result.fetchone()

            if row is None:
                return None

            return {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2],
                "type": row[3],
                "dashboard_state": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "user_id": row[7],
            }

    async def delete_saved_dashboard(self, user_id: str, dashboard_id: str) -> bool:
        """Delete a saved dashboard."""
        try:
            dashboard_uuid = uuid.UUID(dashboard_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                DELETE FROM saved_dashboards
                WHERE id = %s AND user_id = %s
                """,
                (dashboard_uuid, user_id),
            )
            return result.rowcount > 0

    async def update_saved_dashboard(
        self,
        user_id: str,
        dashboard_id: str,
        name: str,
        description: str | None,
        dashboard_type: str,
        dashboard_state: dict[str, Any],
    ) -> bool:
        """Update an existing saved dashboard."""
        try:
            dashboard_uuid = uuid.UUID(dashboard_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                UPDATE saved_dashboards
                SET name = %s, description = %s, type = %s, dashboard_state = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
                """,
                (name, description, dashboard_type, Jsonb(dashboard_state), dashboard_uuid, user_id),
            )
            return result.rowcount > 0

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
    ) -> dict[str, Any]:
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO eval_tasks (policy_id, sim_suite, attributes)
                VALUES (%s, %s, %s)
                RETURNING id, policy_id, sim_suite, status, assigned_at,
                         assignee, created_at, attributes, retries
                """,
                (policy_id, sim_suite, Jsonb(attributes)),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to create eval task")
            return {
                "id": row[0],
                "policy_id": row[1],
                "sim_suite": row[2],
                "status": row[3],
                "assigned_at": row[4],
                "assignee": row[5],
                "created_at": row[6],
                "attributes": row[7],
                "retries": row[8],
            }

    async def get_available_tasks(self, limit: int = 200) -> list[dict[str, Any]]:
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT et.id, policy_id, sim_suite, status, assigned_at,
                       assignee, et.created_at, attributes, retries, p.name
                FROM eval_tasks et
                JOIN policies p ON et.policy_id = p.id
                WHERE status = 'unprocessed'
                  AND assignee IS NULL
                ORDER BY et.created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await result.fetchall()
            return [
                {
                    "id": row[0],
                    "policy_id": row[1],
                    "sim_suite": row[2],
                    "status": row[3],
                    "assigned_at": row[4],
                    "assignee": row[5],
                    "created_at": row[6],
                    "attributes": row[7],
                    "retries": row[8],
                    "policy_name": row[9],
                }
                for row in rows
            ]

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
                SET assignee = %s, assigned_at = NOW(), retries = retries +1
                WHERE id = ANY(%s)
                    AND status = 'unprocessed'
                    AND assignee IS NULL
                RETURNING id
                """,
                (assignee, task_ids),
            )
            rows = await result.fetchall()
            return [row[0] for row in rows]

    async def get_claimed_tasks(self, assignee: str | None = None) -> list[dict[str, Any]]:
        async with self.connect() as con:
            if assignee is not None:
                result = await con.execute(
                    """
                    SELECT et.id, policy_id, sim_suite, status, assigned_at,
                            assignee, et.created_at, attributes, retries, p.name
                    FROM eval_tasks et
                    JOIN policies p ON et.policy_id = p.id
                    WHERE assignee = %s AND status = 'unprocessed'
                    ORDER BY et.created_at ASC
                    """,
                    (assignee,),
                )
            else:
                result = await con.execute(
                    """
                    SELECT et.id, policy_id, sim_suite, status, assigned_at,
                            assignee, et.created_at, attributes, retries, p.name
                    FROM eval_tasks et
                    JOIN policies p ON et.policy_id = p.id
                    WHERE status = 'unprocessed' AND assignee IS NOT NULL
                    ORDER BY et.created_at ASC
                    """,
                )
            rows = await result.fetchall()
            return [
                {
                    "id": row[0],
                    "policy_id": row[1],
                    "sim_suite": row[2],
                    "status": row[3],
                    "assigned_at": row[4],
                    "assignee": row[5],
                    "created_at": row[6],
                    "attributes": row[7],
                    "retries": row[8],
                    "policy_name": row[9],
                }
                for row in rows
            ]

    async def update_task_statuses(
        self,
        updates: dict[uuid.UUID, TaskStatusUpdate],
        require_assignee: str | None = None,
    ) -> dict[uuid.UUID, str]:
        if not updates:
            return {}

        updated = {}
        async with self.connect() as con:
            for task_id, update in updates.items():
                if not require_assignee:
                    if update.clear_assignee:
                        result = await con.execute(
                            """
                            UPDATE eval_tasks
                            SET status = %s,
                                assignee = NULL,
                                attributes = COALESCE(attributes, '{}'::jsonb) || %s::jsonb
                            WHERE id = %s
                            RETURNING id
                            """,
                            (update.status, json.dumps(update.attributes), task_id),
                        )
                    else:
                        result = await con.execute(
                            """
                            UPDATE eval_tasks
                            SET status = %s,
                                attributes = COALESCE(attributes, '{}'::jsonb) || %s::jsonb
                            WHERE id = %s
                            RETURNING id
                            """,
                            (update.status, json.dumps(update.attributes), task_id),
                        )
                else:
                    if update.clear_assignee:
                        result = await con.execute(
                            """
                            UPDATE eval_tasks
                            SET status = %s,
                                assignee = NULL,
                                attributes = COALESCE(attributes, '{}'::jsonb) || %s::jsonb
                            WHERE id = %s AND assignee = %s
                            RETURNING id
                            """,
                            (update.status, json.dumps(update.attributes), task_id, require_assignee),
                        )
                    else:
                        result = await con.execute(
                            """
                            UPDATE eval_tasks
                            SET status = %s,
                                attributes = COALESCE(attributes, '{}'::jsonb) || %s::jsonb
                            WHERE id = %s AND assignee = %s
                            RETURNING id
                            """,
                            (update.status, json.dumps(update.attributes), task_id, require_assignee),
                        )
                if result.rowcount > 0:
                    updated[task_id] = update.status

        return updated

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

    async def get_sweep_by_name(self, name: str) -> dict[str, Any] | None:
        """Get sweep by name."""
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT id, name, project, entity, wandb_sweep_id, state, run_counter,
                       user_id, created_at, updated_at
                FROM sweeps
                WHERE name = %s
                """,
                (name,),
            )
            row = await result.fetchone()
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "name": row[1],
                "project": row[2],
                "entity": row[3],
                "wandb_sweep_id": row[4],
                "state": row[5],
                "run_counter": row[6],
                "user_id": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }

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

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> dict[str, Any] | None:
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT et.id, policy_id, sim_suite, status, assigned_at,
                       assignee, et.created_at, attributes, retries, p.name
                FROM eval_tasks et
                JOIN policies p ON et.policy_id = p.id
                WHERE assignee = %s
                  AND assigned_at IS NOT NULL
                ORDER BY assigned_at DESC
                LIMIT 1
                """,
                (assignee,),
            )
            row = await result.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "policy_id": row[1],
                "sim_suite": row[2],
                "status": row[3],
                "assigned_at": row[4],
                "assignee": row[5],
                "created_at": row[6],
                "attributes": row[7],
                "retries": row[8],
                "policy_name": row[9],
            }

    async def get_all_tasks(self, limit: int = 500) -> list[dict[str, Any]]:
        async with self.connect() as con:
            result = await con.execute(
                """
                SELECT et.id, et.policy_id, et.sim_suite, et.status, et.assigned_at,
                       et.assignee, et.created_at, et.attributes, et.retries,
                       p.name as policy_name
                FROM eval_tasks et
                LEFT JOIN policies p ON et.policy_id = p.id
                ORDER BY et.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await result.fetchall()
            return [
                {
                    "id": row[0],
                    "policy_id": row[1],
                    "sim_suite": row[2],
                    "status": row[3],
                    "assigned_at": row[4],
                    "assignee": row[5],
                    "created_at": row[6],
                    "attributes": row[7],
                    "retries": row[8],
                    "policy_name": row[9],
                }
                for row in rows
            ]
