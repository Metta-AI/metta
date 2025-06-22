import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from psycopg import Connection
from psycopg.types.json import Jsonb

from app_backend.schema_manager import SqlMigration, run_migrations

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
]


class MettaRepo:
    def __init__(self, db_uri: str) -> None:
        self.db_uri = db_uri
        with Connection.connect(self.db_uri) as con:
            run_migrations(con, MIGRATIONS)

    def connect(self) -> Connection:
        return Connection.connect(self.db_uri)

    def get_policy_ids(self, policy_names: List[str]) -> Dict[str, uuid.UUID]:
        if not policy_names:
            return {}

        with self.connect() as con:
            res = con.execute(
                """
                SELECT id, name FROM policies WHERE name = ANY(%s)
                """,
                (policy_names,),
            ).fetchall()
            return {row[1]: row[0] for row in res}

    def create_training_run(self, name: str, user_id: str, attributes: Dict[str, str], url: str | None) -> uuid.UUID:
        status = "running"
        with self.connect() as con:
            result = con.execute(
                """
                INSERT INTO training_runs (name, user_id, attributes, status, url)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, user_id, Jsonb(attributes), status, url),
            ).fetchone()
            if result is None:
                raise RuntimeError("Failed to insert training run")
            return result[0]

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: Dict[str, str],
    ) -> uuid.UUID:
        with self.connect() as con:
            result = con.execute(
                """
                INSERT INTO epochs (run_id, start_training_epoch, end_training_epoch, attributes)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, start_training_epoch, end_training_epoch, Jsonb(attributes)),
            ).fetchone()
            if result is None:
                raise RuntimeError("Failed to insert policy epoch")
            return result[0]

    def create_policy(
        self,
        name: str,
        description: str | None,
        url: str | None,
        epoch_id: uuid.UUID | None,
    ) -> uuid.UUID:
        with self.connect() as con:
            result = con.execute(
                """
                INSERT INTO policies (name, description, url, epoch_id) VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, description, url, epoch_id),
            ).fetchone()
            if result is None:
                raise RuntimeError("Failed to insert policy")
            return result[0]

    def record_episode(
        self,
        agent_policies: Dict[int, uuid.UUID],
        agent_metrics: Dict[int, Dict[str, float]],
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None,
        eval_name: str | None,
        simulation_suite: str | None,
        replay_url: str | None,
        attributes: Dict[str, Any],
    ) -> uuid.UUID:
        with self.connect() as con:
            # Insert into episodes table
            result = con.execute(
                """
                INSERT INTO episodes (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    primary_policy_id,
                    stats_epoch,
                    attributes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                ) RETURNING id
                """,
                (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    primary_policy_id,
                    stats_epoch,
                    Jsonb(attributes),
                ),
            ).fetchone()
            if result is None:
                raise RuntimeError("Failed to insert episode record")
            episode_id = result[0]

            # Insert agent policies
            for agent_id, policy_id in agent_policies.items():
                con.execute(
                    """
                    INSERT INTO episode_agent_policies (
                        episode_id,
                        policy_id,
                        agent_id
                    ) VALUES (%s, %s, %s)
                    """,
                    (episode_id, policy_id, agent_id),
                )

            # Insert agent metrics
            for agent_id, metrics in agent_metrics.items():
                for metric_name, value in metrics.items():
                    con.execute(
                        """
                        INSERT INTO episode_agent_metrics (
                            episode_id,
                            agent_id,
                            metric,
                            value
                        ) VALUES (%s, %s, %s, %s)
                        """,
                        (episode_id, agent_id, metric_name, value),
                    )

            return episode_id

    def get_suites(self) -> List[str]:
        with self.connect() as con:
            result = con.execute("""
                SELECT DISTINCT simulation_suite
                FROM episodes
                WHERE simulation_suite IS NOT NULL
                ORDER BY simulation_suite
            """)
            return [row[0] for row in result]

    def get_metrics(self, suite: str) -> List[str]:
        """Get all available metrics for a given suite."""
        with self.connect() as con:
            result = con.execute(
                """
                SELECT DISTINCT eam.metric
                FROM episodes e
                JOIN episode_agent_metrics eam ON e.id = eam.episode_id
                WHERE e.simulation_suite = %s
                ORDER BY eam.metric
            """,
                (suite,),
            )
            return [row[0] for row in result]

    def get_group_ids(self, suite: str) -> List[str]:
        """Get all available group IDs for a given suite."""
        with self.connect() as con:
            result = con.execute(
                """
                SELECT DISTINCT jsonb_object_keys(e.attributes->'agent_groups') as group_id
                FROM episodes e
                WHERE e.simulation_suite = %s
                ORDER BY group_id
            """,
                (suite,),
            )
            return [row[0] for row in result]

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def create_machine_token(self, user_id: str, name: str, expiration_days: int = 365) -> str:
        """Create a new machine token for a user."""
        # Generate a secure random token
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token)

        # Set expiration time
        expiration_time = datetime.now() + timedelta(days=expiration_days)

        with self.connect() as con:
            con.execute(
                """
                INSERT INTO machine_tokens (user_id, name, token_hash, expiration_time)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, name, token_hash, expiration_time),
            )

        return token

    def list_machine_tokens(self, user_id: str) -> List[Dict[str, Any]]:
        """List all machine tokens for a user."""
        with self.connect() as con:
            result = con.execute(
                """
                SELECT id, name, created_at, expiration_time, last_used_at
                FROM machine_tokens
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "created_at": row[2],
                    "expiration_time": row[3],
                    "last_used_at": row[4],
                }
                for row in result
            ]

    def delete_machine_token(self, user_id: str, token_id: str) -> bool:
        """Delete a machine token."""
        try:
            token_uuid = uuid.UUID(token_id)
        except ValueError:
            return False

        with self.connect() as con:
            result = con.execute(
                """
                DELETE FROM machine_tokens
                WHERE id = %s AND user_id = %s
                """,
                (token_uuid, user_id),
            )
            return result.rowcount > 0

    def validate_machine_token(self, token: str) -> str | None:
        """Validate a machine token and return the user_id if valid."""
        token_hash = self._hash_token(token)

        with self.connect() as con:
            result = con.execute(
                """
                UPDATE machine_tokens
                SET last_used_at = CURRENT_TIMESTAMP
                WHERE token_hash = %s AND expiration_time > CURRENT_TIMESTAMP
                RETURNING user_id
                """,
                (token_hash,),
            ).fetchone()

            if result:
                return result[0]
            return None
