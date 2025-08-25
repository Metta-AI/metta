import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client

# Import TrainingRunRow from metta_repo for now
from metta.app_backend.metta_repo import TrainingRunRow

logger = logging.getLogger(__name__)


class StatsRepo:
    """ClickHouse-based repository for stats data."""

    def __init__(self, clickhouse_uri: str):
        """Initialize with ClickHouse connection URI.

        Expected format: clickhouse://username:password@host:port/database
        """
        self._parse_uri(clickhouse_uri)
        self._client = None
        self._initialized = False

    def _parse_uri(self, uri: str) -> None:
        """Parse ClickHouse URI into connection parameters."""
        if not uri.startswith("clickhouse://"):
            raise ValueError("ClickHouse URI must start with clickhouse://")

        # Remove protocol
        uri = uri[13:]  # len("clickhouse://")

        # Default values
        self.username = "default"
        self.password = ""
        self.host = "localhost"
        self.port = 0

        # Parse auth and host
        if "@" in uri:
            auth, host_part = uri.split("@", 1)
            if ":" in auth:
                self.username, self.password = auth.split(":", 1)
            else:
                self.username = auth
        else:
            host_part = uri

        # Parse host, port, database
        if "/" in host_part:
            host_port, self.database = host_part.split("/", 1)
        else:
            host_port = host_part

        if ":" in host_port:
            self.host, port_str = host_port.split(":", 1)
            self.port = int(port_str)
        else:
            self.host = host_port

    def _get_client(self) -> Client:
        """Get or create ClickHouse client."""
        if self._client is None:
            try:
                self._client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    secure=True,
                )
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse at {self.host}:{self.port}: {e}")
                raise
        return self._client

    async def _ensure_tables_exist(self) -> None:
        """Ensure all required ClickHouse tables exist."""
        if self._initialized:
            return

        client = self._get_client()

        # Create tables with ClickHouse-optimized schemas
        tables = [
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id UUID DEFAULT generateUUIDv4(),
                name String,
                created_at DateTime64(3) DEFAULT now64(),
                user_id String,
                finished_at Nullable(DateTime64(3)),
                status String,
                url Nullable(String),
                description Nullable(String),
                tags Array(String) DEFAULT [],
                attributes String DEFAULT '{}'  -- JSON as string
            ) ENGINE = MergeTree()
            ORDER BY (user_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS epochs (
                id UUID DEFAULT generateUUIDv4(),
                run_id UUID,
                start_training_epoch UInt32,
                end_training_epoch UInt32,
                created_at DateTime64(3) DEFAULT now64(),
                attributes String DEFAULT '{}'  -- JSON as string
            ) ENGINE = MergeTree()
            ORDER BY (run_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS policies (
                id UUID DEFAULT generateUUIDv4(),
                name String,
                description Nullable(String),
                url Nullable(String),
                created_at DateTime64(3) DEFAULT now64(),
                epoch_id Nullable(UUID)
            ) ENGINE = MergeTree()
            ORDER BY (name, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id UUID DEFAULT generateUUIDv4(),
                internal_id UInt64 DEFAULT 0,  -- For compatibility, but not used in ClickHouse
                created_at DateTime64(3) DEFAULT now64(),
                primary_policy_id UUID,
                stats_epoch Nullable(UUID),
                replay_url Nullable(String),
                eval_name Nullable(String),
                simulation_suite Nullable(String),
                eval_category Nullable(String),
                env_name String,
                attributes String DEFAULT '{}',  -- JSON as string
                eval_task_id Nullable(UUID),
                thumbnail_url Nullable(String)
            ) ENGINE = MergeTree()
            ORDER BY (primary_policy_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS episode_agent_policies (
                episode_id UUID,
                policy_id UUID,
                agent_id UInt32
            ) ENGINE = MergeTree()
            ORDER BY (episode_id, policy_id, agent_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS episode_agent_metrics (
                episode_id UUID,
                agent_id UInt32,
                metric String,
                value Float64
            ) ENGINE = MergeTree()
            ORDER BY (episode_id, agent_id, metric)
            """,
            """
            CREATE TABLE IF NOT EXISTS episode_tags (
                episode_id UUID,
                tag String
            ) ENGINE = MergeTree()
            ORDER BY (episode_id, tag)
            """,
        ]

        for table_sql in tables:
            client.command(table_sql)

        self._initialized = True

    @asynccontextmanager
    async def connect(self):
        """Context manager for database connections. Returns synchronous client."""
        await self._ensure_tables_exist()
        client = self._get_client()
        try:
            yield client
        finally:
            # clickhouse-connect handles connection pooling internally
            pass

    async def get_policy_ids(self, policy_names: list[str]) -> dict[str, uuid.UUID]:
        """Get policy IDs for given policy names."""
        if not policy_names:
            return {}

        async with self.connect() as client:
            # Use parameterized query for safety
            query = "SELECT toString(id), name FROM policies WHERE name IN {policy_names:Array(String)}"

            result = client.query(query, {"policy_names": policy_names})
            return {row[1]: uuid.UUID(row[0]) for row in result.result_rows}

    async def create_training_run(
        self,
        name: str,
        user_id: str,
        attributes: dict[str, str],
        url: str | None,
        description: str | None,
        tags: list[str] | None,
    ) -> uuid.UUID:
        """Create a new training run."""
        import json

        run_id = uuid.uuid4()
        status = "running"
        tags = tags or []

        async with self.connect() as client:
            # Check if training run already exists
            existing = client.query(
                "SELECT id FROM training_runs WHERE user_id = {user_id:String} AND name = {name:String}",
                {"user_id": user_id, "name": name},
            )

            if existing.result_rows:
                return uuid.UUID(existing.result_rows[0][0])

            # Insert new training run
            client.insert(
                "training_runs",
                [
                    [
                        str(run_id),
                        name,
                        datetime.now(),
                        user_id,
                        None,  # finished_at
                        status,
                        url,
                        description,
                        tags,
                        json.dumps(attributes),
                    ]
                ],
                column_names=[
                    "id",
                    "name",
                    "created_at",
                    "user_id",
                    "finished_at",
                    "status",
                    "url",
                    "description",
                    "tags",
                    "attributes",
                ],
            )

            return run_id

    async def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        """Update the status of a training run."""
        async with self.connect() as client:
            client.command(
                """
                ALTER TABLE training_runs UPDATE
                status = {status:String},
                finished_at = CASE WHEN {status:String} != 'running' THEN now64() ELSE finished_at END
                WHERE id = {run_id:String}
                """,
                {"status": status, "run_id": str(run_id)},
            )

            # ClickHouse doesn't return rowcount like PostgreSQL, so we'll check if the record exists
            check = client.query(
                "SELECT count() FROM training_runs WHERE id = {run_id:String}", {"run_id": str(run_id)}
            )

            if check.result_rows[0][0] == 0:
                raise ValueError(f"Training run with ID {run_id} not found")

    async def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, str],
    ) -> uuid.UUID:
        """Create a new policy epoch."""
        import json

        epoch_id = uuid.uuid4()

        async with self.connect() as client:
            client.insert(
                "epochs",
                [
                    [
                        str(epoch_id),
                        str(run_id),
                        start_training_epoch,
                        end_training_epoch,
                        datetime.now(),
                        json.dumps(attributes),
                    ]
                ],
                column_names=["id", "run_id", "start_training_epoch", "end_training_epoch", "created_at", "attributes"],
            )

            return epoch_id

    async def create_policy(
        self,
        name: str,
        description: str | None,
        url: str | None,
        epoch_id: uuid.UUID | None,
    ) -> uuid.UUID:
        """Create a new policy."""
        policy_id = uuid.uuid4()

        async with self.connect() as client:
            client.insert(
                "policies",
                [
                    [
                        str(policy_id),
                        name,
                        description,
                        url,
                        datetime.now(),
                        str(epoch_id) if epoch_id else None,
                    ]
                ],
                column_names=["id", "name", "description", "url", "created_at", "epoch_id"],
            )

            return policy_id

    async def record_episode(
        self,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None,
        sim_name: str,
        env_label: str,
        replay_url: str | None,
        attributes: dict[str, Any],
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> uuid.UUID:
        """Record a new episode with agent policies and metrics."""
        import json

        def _get_simulation_suite(sim_name: str) -> str:
            for delim in ["/", "."]:
                if delim in sim_name:
                    return sim_name.split(delim)[0]
            return sim_name

        episode_id = uuid.uuid4()
        simulation_suite = _get_simulation_suite(sim_name) if sim_name else None

        async with self.connect() as client:
            # Insert into episodes table
            client.insert(
                "episodes",
                [
                    [
                        str(episode_id),
                        0,  # internal_id - not used in ClickHouse
                        datetime.now(),
                        str(primary_policy_id),
                        str(stats_epoch) if stats_epoch else None,
                        replay_url,
                        sim_name,
                        simulation_suite,
                        simulation_suite,  # eval_category same as simulation_suite
                        env_label,
                        json.dumps(attributes),
                        str(eval_task_id) if eval_task_id else None,
                        thumbnail_url,
                    ]
                ],
                column_names=[
                    "id",
                    "internal_id",
                    "created_at",
                    "primary_policy_id",
                    "stats_epoch",
                    "replay_url",
                    "eval_name",
                    "simulation_suite",
                    "eval_category",
                    "env_name",
                    "attributes",
                    "eval_task_id",
                    "thumbnail_url",
                ],
            )

            # Insert agent policies
            if agent_policies:
                policy_rows = [
                    [str(episode_id), str(policy_id), agent_id] for agent_id, policy_id in agent_policies.items()
                ]
                client.insert(
                    "episode_agent_policies", policy_rows, column_names=["episode_id", "policy_id", "agent_id"]
                )

            # Insert agent metrics
            if agent_metrics:
                metric_rows = [
                    [str(episode_id), agent_id, metric_name, value]
                    for agent_id, metrics in agent_metrics.items()
                    for metric_name, value in metrics.items()
                ]
                client.insert(
                    "episode_agent_metrics", metric_rows, column_names=["episode_id", "agent_id", "metric", "value"]
                )

            # Insert episode tags
            if tags:
                tag_rows = [[str(episode_id), tag] for tag in tags]
                client.insert("episode_tags", tag_rows, column_names=["episode_id", "tag"])

            return episode_id

    async def get_training_runs(self) -> list[TrainingRunRow]:
        """Get all training runs."""
        async with self.connect() as client:
            result = client.query("""
                SELECT
                    toString(id) as id,
                    name,
                    user_id,
                    created_at,
                    finished_at,
                    status,
                    url,
                    description,
                    tags,
                    attributes
                FROM training_runs
                ORDER BY created_at DESC
            """)

            return [
                TrainingRunRow(
                    id=uuid.UUID(row[0]),
                    name=row[1],
                    user_id=row[2],
                    created_at=row[3],
                    finished_at=row[4],
                    status=row[5],
                    url=row[6],
                    description=row[7],
                    tags=row[8] or [],
                    attributes=row[9] or {},
                )
                for row in result.result_rows
            ]

    async def get_training_run(self, run_id: str) -> TrainingRunRow | None:
        """Get a specific training run by ID."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return None

        async with self.connect() as client:
            result = client.query(
                """
                SELECT
                    toString(id) as id,
                    name,
                    user_id,
                    created_at,
                    finished_at,
                    status,
                    url,
                    description,
                    tags,
                    attributes
                FROM training_runs
                WHERE id = {run_id:String}
                LIMIT 1
            """,
                {"run_id": str(run_uuid)},
            )

            if not result.result_rows:
                return None

            row = result.result_rows[0]
            return TrainingRunRow(
                id=uuid.UUID(row[0]),
                name=row[1],
                user_id=row[2],
                created_at=row[3],
                finished_at=row[4],
                status=row[5],
                url=row[6],
                description=row[7],
                tags=row[8] or [],
                attributes=row[9] or {},
            )

    async def update_training_run_description(self, user_id: str, run_id: str, description: str) -> bool:
        """Update the description of a training run."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return False

        async with self.connect() as client:
            # First check if the run exists and belongs to the user
            check_result = client.query(
                """
                SELECT COUNT(*) FROM training_runs
                WHERE id = {run_id:String} AND user_id = {user_id:String}
            """,
                {"run_id": str(run_uuid), "user_id": user_id},
            )

            if not check_result.result_rows or check_result.result_rows[0][0] == 0:
                return False

            # Update the description
            client.command(
                """
                ALTER TABLE training_runs
                UPDATE description = {description:String}
                WHERE id = {run_id:String} AND user_id = {user_id:String}
            """,
                {"description": description, "run_id": str(run_uuid), "user_id": user_id},
            )

            return True

    async def update_training_run_tags(self, user_id: str, run_id: str, tags: list[str]) -> bool:
        """Update the tags of a training run."""
        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError:
            return False

        async with self.connect() as client:
            # First check if the run exists and belongs to the user
            check_result = client.query(
                """
                SELECT COUNT(*) FROM training_runs
                WHERE id = {run_id:String} AND user_id = {user_id:String}
            """,
                {"run_id": str(run_uuid), "user_id": user_id},
            )

            if not check_result.result_rows or check_result.result_rows[0][0] == 0:
                return False

            # Update the tags
            client.command(
                """
                ALTER TABLE training_runs
                UPDATE tags = {tags:Array(String)}
                WHERE id = {run_id:String} AND user_id = {user_id:String}
            """,
                {"tags": tags, "run_id": str(run_uuid), "user_id": user_id},
            )

            return True

    async def get_policies(self) -> list:
        """Get all policies from ClickHouse."""
        async with self.connect() as client:
            result = client.query("""
                SELECT
                    toString(id) as id,
                    'policy' as type,
                    name,
                    created_at,
                    description,
                    url,
                    epoch_id
                FROM policies
                ORDER BY created_at DESC
            """)

            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "user_id": None,  # Policies don't have user_id in ClickHouse
                    "created_at": str(row[3]),
                    "tags": [],  # Policies don't have tags in ClickHouse yet
                    "description": row[4],
                    "url": row[5],
                    "epoch_id": str(row[6]) if row[6] else None,
                }
                for row in result.result_rows
            ]

    async def get_eval_names_for_selection(
        self, training_run_ids: list[str], run_free_policy_ids: list[str]
    ) -> list[str]:
        """Get distinct evaluation names for selected training runs and policies."""
        if not training_run_ids and not run_free_policy_ids:
            return []

        async with self.connect() as client:
            conditions = []

            # Build conditions for training run policies
            if training_run_ids:
                # Get episodes from policies that belong to the specified training runs
                conditions.append("""
                    episodes.primary_policy_id IN (
                        SELECT toString(p.id)
                        FROM policies p
                        JOIN epochs e ON p.epoch_id = e.id
                        WHERE toString(e.run_id) IN {training_run_ids:Array(String)}
                    )
                """)

            # Build conditions for run-free policies
            if run_free_policy_ids:
                conditions.append("""
                    episodes.primary_policy_id IN {run_free_policy_ids:Array(String)}
                """)

            # Combine conditions
            where_clause = " OR ".join(f"({condition})" for condition in conditions)

            query = f"""
                SELECT DISTINCT episodes.eval_name
                FROM episodes
                WHERE ({where_clause})
                AND episodes.eval_name IS NOT NULL
                ORDER BY episodes.eval_name
            """

            params = {}
            if training_run_ids:
                params["training_run_ids"] = training_run_ids
            if run_free_policy_ids:
                params["run_free_policy_ids"] = run_free_policy_ids

            result = client.query(query, params)
            return [row[0] for row in result.result_rows]

    async def get_available_metrics_for_selection(
        self, training_run_ids: list[str], run_free_policy_ids: list[str], eval_names: list[str]
    ) -> list[str]:
        """Get available metrics for selected training runs, policies and evaluations."""
        if not training_run_ids and not run_free_policy_ids:
            return []
        if not eval_names:
            return []

        async with self.connect() as client:
            conditions = []

            # Build conditions for training run policies
            if training_run_ids:
                conditions.append("""
                    episodes.primary_policy_id IN (
                        SELECT toString(p.id)
                        FROM policies p
                        JOIN epochs e ON p.epoch_id = e.id
                        WHERE toString(e.run_id) IN {training_run_ids:Array(String)}
                    )
                """)

            # Build conditions for run-free policies
            if run_free_policy_ids:
                conditions.append("""
                    episodes.primary_policy_id IN {run_free_policy_ids:Array(String)}
                """)

            # Combine conditions
            where_clause = " OR ".join(f"({condition})" for condition in conditions)

            query = f"""
                SELECT DISTINCT eam.metric
                FROM episode_agent_metrics eam
                JOIN episodes ON episodes.id = eam.episode_id
                WHERE ({where_clause})
                AND episodes.eval_name IN {{eval_names:Array(String)}}
                ORDER BY eam.metric
            """

            params = {"eval_names": eval_names}
            if training_run_ids:
                params["training_run_ids"] = training_run_ids
            if run_free_policy_ids:
                params["run_free_policy_ids"] = run_free_policy_ids

            result = client.query(query, params)
            return [row[0] for row in result.result_rows]

    async def fetch_policy_scorecard_data(
        self, training_run_ids: list[str], run_free_policy_ids: list[str], eval_names: list[str], metric: str
    ) -> list[dict]:
        """Fetch evaluation data for policy-based scorecard."""
        if not eval_names or not metric:
            return []
        if not training_run_ids and not run_free_policy_ids:
            return []

        async with self.connect() as client:
            conditions = []

            # Build conditions for training run policies
            if training_run_ids:
                conditions.append("""
                    episodes.primary_policy_id IN (
                        SELECT p.id
                        FROM policies p
                        JOIN epochs e ON p.epoch_id = e.id
                        WHERE e.run_id IN {training_run_ids:Array(String)}
                    )
                """)

            # Build conditions for run-free policies
            if run_free_policy_ids:
                conditions.append("""
                    episodes.primary_policy_id IN {run_free_policy_ids:Array(String)}
                """)

            # Combine conditions
            where_clause = " OR ".join(f"({condition})" for condition in conditions)

            query = f"""
                SELECT
                    episodes.primary_policy_id as policy_id,
                    policies.name as policy_name,
                    episodes.eval_category,
                    episodes.env_name,
                    any(episodes.replay_url) as replay_url,
                    any(episodes.thumbnail_url) as thumbnail_url,
                    sum(eam.value) as total_score,
                    count() as num_agents,
                    max(episodes.id) as episode_id,
                    epochs.run_id as run_id,
                    epochs.end_training_epoch as epoch
                FROM episodes
                JOIN episode_agent_metrics eam ON episodes.id = eam.episode_id
                JOIN policies ON episodes.primary_policy_id = policies.id
                LEFT JOIN epochs ON policies.epoch_id = epochs.id
                WHERE ({where_clause})
                AND eam.metric = {{metric:String}}
                AND episodes.eval_name IN {{eval_names:Array(String)}}
                GROUP BY episodes.primary_policy_id, policies.name, episodes.eval_category,
                  episodes.env_name, epochs.run_id, epochs.end_training_epoch
                ORDER BY policies.name, episodes.eval_category, episodes.env_name
            """

            params = {"metric": metric, "eval_names": eval_names}
            if training_run_ids:
                params["training_run_ids"] = training_run_ids  # Keep as strings, ClickHouse will convert
            if run_free_policy_ids:
                params["run_free_policy_ids"] = run_free_policy_ids  # Keep as strings, ClickHouse will convert

            result = client.query(query, params)

            # Convert to list of dictionaries matching PolicyEvaluationResult structure
            return [
                {
                    "policy_id": str(row[0]),  # Convert UUID to string
                    "policy_name": row[1],
                    "eval_category": row[2],
                    "env_name": row[3],
                    "replay_url": row[4],
                    "thumbnail_url": row[5],
                    "total_score": float(row[6]),
                    "num_agents": int(row[7]),
                    "episode_id": int(str(row[8])[-8:], 16)
                    if row[8]
                    else 0,  # Extract some digits from UUID for episode_id
                    "run_id": str(row[9])
                    if row[9] and str(row[9]) != "00000000-0000-0000-0000-000000000000"
                    else None,  # Treat zero UUID as None
                    "epoch": int(row[10]) if row[10] is not None else None,
                }
                for row in result.result_rows
            ]

    async def get_policy_metric_stats(
        self, policy_ids: list[str], eval_names: list[str], metrics: list[str]
    ) -> list[tuple]:
        """Get aggregated metric statistics (min, max, avg) for policies, evaluations, and metrics."""
        if not policy_ids or not eval_names or not metrics:
            return []

        async with self.connect() as client:
            query = """
                SELECT
                    episodes.primary_policy_id as policy_id,
                    episodes.eval_name as eval_name,
                    eam.metric as metric,
                    min(eam.value) as min_value,
                    max(eam.value) as max_value,
                    avg(eam.value) as avg_value
                FROM episode_agent_metrics eam
                JOIN episodes ON episodes.id = eam.episode_id
                WHERE
                    episodes.primary_policy_id IN {policy_ids:Array(String)}
                    AND episodes.eval_name IN {eval_names:Array(String)}
                    AND eam.metric IN {metrics:Array(String)}
                GROUP BY episodes.primary_policy_id, episodes.eval_name, eam.metric
                ORDER BY episodes.primary_policy_id, episodes.eval_name, eam.metric
            """

            result = client.query(query, {"policy_ids": policy_ids, "eval_names": eval_names, "metrics": metrics})

            # Return tuples in the format expected by fetch_policy_scores
            return [
                (
                    str(row[0]),  # policy_id as string
                    row[1],  # eval_name
                    row[2],  # metric
                    float(row[3]),  # min_value
                    float(row[4]),  # max_value
                    float(row[5]),  # avg_value
                )
                for row in result.result_rows
            ]
