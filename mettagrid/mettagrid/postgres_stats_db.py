from typing import Any, Dict, List, LiteralString, Sequence, Tuple

from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from metta.util.schema_manager import SqlMigration, run_migrations

# This is a list of migrations that will be applied to the eval database.
# Do not change existing migrations, only add new ones.
EVAL_MIGRATIONS = [
    SqlMigration(
        version=0,
        description="Initial eval schema",
        sql_statements=[
            """CREATE TABLE training_runs (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                finished_at TIMESTAMP,
                status TEXT NOT NULL,
                url TEXT,
                attributes JSONB
            )""",
            """CREATE TABLE policy_epochs (
                id SERIAL PRIMARY KEY,
                run_id INTEGER NOT NULL,
                start_training_epoch INTEGER NOT NULL,
                end_training_epoch INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                attributes JSONB
            )""",
            """CREATE TABLE policies (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                url TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                epoch_id INTEGER REFERENCES policy_epochs(id),
                UNIQUE (name)
            )""",
            # This is slightly denormalized, in the sense that it is storing both the attributes of the env and
            # the attributes of the episode. We could imagine having separate (env, env_attributes) tables and having
            # a foreign key into envs in episodes.  However, I can imagine a lot of envs that are only used in a single
            # episode, so I'm not sure it's worth the extra complexity.
            """CREATE TABLE episodes (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                replay_url TEXT,
                eval_name TEXT,
                simulation_suite TEXT,
                attributes JSONB
            )""",
            """CREATE TABLE episode_agent_policies (
                episode_id INTEGER NOT NULL REFERENCES episodes(id),
                policy_id INTEGER NOT NULL REFERENCES policies(id),
                agent_id INTEGER NOT NULL,
                PRIMARY KEY (episode_id, policy_id, agent_id)
            )""",
            """CREATE TABLE episode_agent_metrics (
                episode_id INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL,
                PRIMARY KEY (episode_id, agent_id, metric)
            )""",
        ],
    ),
]


class PostgresStatsDB:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.con = Connection.connect(self.db_url)
        run_migrations(self.con, EVAL_MIGRATIONS)

    def __enter__(self) -> "PostgresStatsDB":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.con.close()

    def query(self, query: LiteralString, params: Tuple[Any, ...] = ()) -> Sequence[TupleRow]:
        with self.con.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def execute(self, query: LiteralString, params: Tuple[Any, ...] = ()) -> None:
        with self.con.cursor() as cur:
            cur.execute(query, params)
            self.con.commit()

    def get_policy_ids(self, policy_names: List[str]) -> Dict[str, int]:
        if not policy_names:
            return {}

        with self.con.cursor() as cur:
            # Use psycopg's built-in list handling
            cur.execute(
                """
                SELECT id, name FROM policies WHERE name = ANY(%s)
                """,
                (policy_names,),
            )
            return {row[1]: row[0] for row in cur.fetchall()}

    def create_training_run(self, name: str, user_id: str, attributes: Dict[str, str], url: str | None) -> int:
        status = "running"
        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (name, user_id, attributes, status, url)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, user_id, Jsonb(attributes), status, url),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert training run")
            self.con.commit()
            return result[0]

    def create_policy_epoch(
        self, run_id: int, start_training_epoch: int, end_training_epoch: int, attributes: Dict[str, str]
    ) -> int:
        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO policy_epochs (run_id, start_training_epoch, end_training_epoch, attributes)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, start_training_epoch, end_training_epoch, Jsonb(attributes)),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert policy epoch")
            self.con.commit()
            return result[0]

    def create_policy(self, name: str, description: str | None, url: str | None, epoch_id: int | None) -> int:
        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO policies (name, description, url, epoch_id) VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, description, url, epoch_id),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert policy")
            self.con.commit()
            return result[0]

    def record_episode(
        self,
        agent_policies: Dict[int, int],
        agent_metrics: Dict[int, Dict[str, float]],
        eval_name: str | None,
        simulation_suite: str | None,
        replay_url: str | None,
        attributes: Dict[str, Any],
    ) -> None:
        with self.con.cursor() as cur:
            # Insert into episodes table
            cur.execute(
                """
                INSERT INTO episodes (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    attributes
                ) VALUES (
                    %s, %s, %s, %s
                ) RETURNING id
                """,
                (
                    replay_url,
                    eval_name,
                    simulation_suite,
                    Jsonb(attributes),
                ),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert episode record")
            episode_id = result[0]

            # Insert agent policies
            for agent_id, policy_id in agent_policies.items():
                cur.execute(
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
                    cur.execute(
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

            self.con.commit()
