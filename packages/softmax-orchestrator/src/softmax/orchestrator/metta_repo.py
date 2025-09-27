import hashlib
import logging
import secrets
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Literal

from psycopg import AsyncConnection, Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import BaseModel, Field

from softmax.orchestrator.query_logger import execute_single_row_query_and_log
from softmax.orchestrator.schema_manager import SqlMigration, run_migrations

TaskStatus = Literal["unprocessed", "canceled", "done", "error"]


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    clear_assignee: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


# Row models for database tables
class SavedDashboardRow(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    type: str
    dashboard_state: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    user_id: str


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


class LeaderboardRow(BaseModel):
    id: uuid.UUID
    name: str
    user_id: str
    evals: list[str]
    metric: str
    start_date: date
    latest_episode: int
    created_at: datetime
    updated_at: datetime


class PolicyEval(BaseModel):
    num_agents: int
    total_score: float


class LeaderboardPolicyScore(BaseModel):
    leaderboard_id: uuid.UUID
    policy_id: uuid.UUID
    score: float


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
        description="Add scorecard performance indexes",
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
    SqlMigration(
        version=20,
        description="Add user_id to eval_tasks",
        sql_statements=[
            """ALTER TABLE eval_tasks ADD COLUMN user_id TEXT""",
            """CREATE INDEX idx_eval_tasks_user_id ON eval_tasks(user_id)""",
        ],
    ),
    SqlMigration(
        version=21,
        description="Add updated_at to eval_tasks",
        sql_statements=[
            """ALTER TABLE eval_tasks ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP""",
            """CREATE INDEX idx_eval_tasks_updated_at ON eval_tasks(updated_at)""",
        ],
    ),
    SqlMigration(
        version=22,
        description="Add index on episodes.primary_policy_id",
        sql_statements=[
            """CREATE INDEX IF NOT EXISTS idx_episodes_primary_policy_id ON episodes(primary_policy_id)""",
        ],
    ),
    SqlMigration(
        version=23,
        description="Add leaderboards table",
        sql_statements=[
            """CREATE TABLE leaderboards (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                evals TEXT[] NOT NULL,
                metric TEXT NOT NULL,
                start_date DATE NOT NULL,
                latest_episode INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE leaderboard_policy_scores (
                leaderboard_id UUID NOT NULL REFERENCES leaderboards(id) ON DELETE CASCADE,
                policy_id UUID NOT NULL REFERENCES policies(id),
                score FLOAT NOT NULL,
                PRIMARY KEY (leaderboard_id, policy_id)
              )""",
            """CREATE OR REPLACE VIEW unified_training_runs AS
                WITH good_policies AS (select distinct primary_policy_id FROM episodes),
                my_training_runs AS (
                  SELECT t.id AS id, 'training_run' AS type, t.name, t.user_id, t.created_at, t.tags
                  FROM training_runs t JOIN epochs e ON t.id = e.run_id
                  JOIN policies p ON p.epoch_id = e.id
                  JOIN good_policies g ON p.id = g.primary_policy_id
                ),
                my_run_free_policies AS (
                  SELECT p.id AS id, 'policy' AS type, p.name, NULL as user_id, p.created_at, NULL::text[] as tags
                  FROM policies p
                  JOIN good_policies g ON p.id = g.primary_policy_id
                  WHERE p.epoch_id IS NULL
                )
                SELECT * FROM my_training_runs
                UNION
                SELECT * FROM my_run_free_policies;""",
        ],
    ),
    SqlMigration(
        version=24,
        description="Convert training_runs.status from TEXT to ENUM",
        sql_statements=[
            # Step 1: Create the ENUM type
            """CREATE TYPE training_run_status AS ENUM ('running', 'completed', 'failed')""",
            # Step 2: Add new column with ENUM type and default
            """ALTER TABLE training_runs ADD COLUMN status_new training_run_status DEFAULT 'running'""",
            # Step 3: Migrate existing data (all should be 'running' currently, but handle edge cases)
            """UPDATE training_runs SET status_new =
                CASE
                    WHEN LOWER(status) = 'running' THEN 'running'::training_run_status
                    WHEN LOWER(status) = 'completed' THEN 'completed'::training_run_status
                    WHEN LOWER(status) = 'failed' THEN 'failed'::training_run_status
                    ELSE 'running'::training_run_status
                END""",
            # Step 4: Make the new column NOT NULL
            """ALTER TABLE training_runs ALTER COLUMN status_new SET NOT NULL""",
            # Step 5: Drop the view that depends on the status column
            """DROP VIEW wide_episodes""",
            # Step 6: Drop old column and rename new one
            """ALTER TABLE training_runs DROP COLUMN status""",
            """ALTER TABLE training_runs RENAME COLUMN status_new TO status""",
            # Step 7: Recreate the view with the new ENUM column
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
    SqlMigration(
        version=25,
        description="Add thumbnail_url field to episodes table and update wide_episodes view",
        sql_statements=[
            """ALTER TABLE episodes ADD COLUMN thumbnail_url TEXT""",
            """DROP VIEW wide_episodes""",
            """CREATE VIEW wide_episodes AS
            SELECT
                e.id,
                e.internal_id,
                e.created_at,
                e.primary_policy_id,
                e.stats_epoch,
                e.replay_url,
                e.thumbnail_url,
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
        version=26,
        description="Drop simulation_suite column from episodes table",
        sql_statements=[
            """DROP VIEW wide_episodes""",
            """ALTER TABLE episodes DROP COLUMN simulation_suite""",
            """CREATE VIEW wide_episodes AS
            SELECT
                e.id,
                e.internal_id,
                e.created_at,
                e.primary_policy_id,
                e.stats_epoch,
                e.replay_url,
                e.thumbnail_url,
                e.eval_name,
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
            logger.error(f"Error connecting to database: {e}. Pool stats: {stats}")

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

    async def list_saved_dashboards(self) -> list[SavedDashboardRow]:
        """List all saved dashboards."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(SavedDashboardRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, description, type, dashboard_state, created_at, updated_at, user_id
                    FROM saved_dashboards
                    ORDER BY updated_at DESC
                    """
                )
                return await cur.fetchall()

    async def get_saved_dashboard(self, dashboard_id: str) -> SavedDashboardRow | None:
        """Get a specific saved dashboard by ID."""
        try:
            dashboard_uuid = uuid.UUID(dashboard_id)
        except ValueError:
            return None

        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(SavedDashboardRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, description, type, dashboard_state, created_at, updated_at, user_id
                    FROM saved_dashboards
                    WHERE id = %s
                    """,
                    (dashboard_uuid,),
                )
                return await cur.fetchone()

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

    async def update_dashboard_state(
        self,
        user_id: str,
        dashboard_id: str,
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
                SET dashboard_state = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
                """,
                (Jsonb(dashboard_state), dashboard_uuid, user_id),
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

    async def create_leaderboard(
        self,
        name: str,
        user_id: str,
        evals: list[str],
        metric: str,
        start_date: str,
    ) -> uuid.UUID:
        """Create a new leaderboard."""
        async with self.connect() as con:
            result = await con.execute(
                """
                INSERT INTO leaderboards (
                    name, user_id, evals, metric, start_date
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    name,
                    user_id,
                    evals,
                    metric,
                    start_date,
                ),
            )
            row = await result.fetchone()
            if row is None:
                raise RuntimeError("Failed to create leaderboard")
            return row[0]

    async def update_leaderboard(
        self,
        leaderboard_id: uuid.UUID,
        user_id: str,
        name: str,
        evals: list[str],
        metric: str,
        start_date: str,
    ) -> LeaderboardRow:
        """Update a leaderboard."""

        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(LeaderboardRow)) as cur:
                cur_leaderboard_cursor = await cur.execute(
                    "SELECT * FROM leaderboards WHERE id = %s AND user_id = %s",
                    (leaderboard_id, user_id),
                )
                cur_leaderboard = await cur_leaderboard_cursor.fetchone()
                if not cur_leaderboard:
                    raise RuntimeError(f"Leaderboard {leaderboard_id} not found or not owned by user {user_id}")

            if (
                cur_leaderboard.name != name
                or cur_leaderboard.evals != evals
                or cur_leaderboard.metric != metric
                or cur_leaderboard.start_date != start_date
            ):
                async with con.cursor(row_factory=class_row(LeaderboardRow)) as update_cur:
                    query = """
                    UPDATE leaderboards
                    SET name = %s, evals = %s, metric = %s, start_date = %s, latest_episode = 0, updated_at = NOW()
                    WHERE id = %s AND user_id = %s
                    RETURNING *
                  """
                    params = (name, evals, metric, start_date, leaderboard_id, user_id)

                    await update_cur.execute(query, params)
                    updated_leaderboard = await update_cur.fetchone()
                    if not updated_leaderboard:
                        raise RuntimeError(f"Leaderboard {leaderboard_id} not found or not owned by user {user_id}")

                    if (
                        cur_leaderboard.evals != evals
                        or cur_leaderboard.metric != metric
                        or cur_leaderboard.start_date != start_date
                    ):
                        # TODO: Technically we shouldn't need to delete the scores if only start_date changes
                        await con.execute(
                            "DELETE FROM leaderboard_policy_scores WHERE leaderboard_id = %s",
                            (leaderboard_id,),
                        )
                    return updated_leaderboard
            else:
                return cur_leaderboard

    async def list_leaderboards(self) -> list[LeaderboardRow]:
        """List all leaderboards for a user."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(LeaderboardRow)) as cur:
                await cur.execute(
                    """
                    SELECT id, name, user_id, evals, metric, start_date, latest_episode, created_at, updated_at
                    FROM leaderboards
                    ORDER BY updated_at DESC
                    """,
                )
                rows = await cur.fetchall()
                return rows

    async def get_leaderboard(self, leaderboard_id: uuid.UUID) -> LeaderboardRow | None:
        """Get a specific leaderboard by ID."""
        async with self.connect() as con:
            async with con.cursor(row_factory=class_row(LeaderboardRow)) as cur:
                await cur.execute("SELECT * FROM leaderboards WHERE id = %s", (leaderboard_id,))
                row = await cur.fetchone()
                return row

    async def delete_leaderboard(self, leaderboard_id: str, user_id: str) -> bool:
        """Delete a leaderboard."""
        try:
            leaderboard_uuid = uuid.UUID(leaderboard_id)
        except ValueError:
            return False

        async with self.connect() as con:
            result = await con.execute(
                """
                DELETE FROM leaderboards
                WHERE id = %s AND user_id = %s
                """,
                (leaderboard_uuid, user_id),
            )
            return result.rowcount > 0

    async def update_leaderboard_latest_episode(
        self, con: AsyncConnection, leaderboard_id: uuid.UUID, latest_episode: int
    ) -> None:
        """Update the latest episode for a leaderboard."""
        await con.execute(
            """
            UPDATE leaderboards SET latest_episode = %s WHERE id = %s
            """,
            (latest_episode, leaderboard_id),
        )
