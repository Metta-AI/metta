from typing import Any, Dict, List, LiteralString, Sequence, Tuple, Union

from sqlalchemy import text
from sqlalchemy.engine import Engine, Row

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


class StatsRepo:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        run_migrations(self.engine, EVAL_MIGRATIONS)

    def __enter__(self) -> "StatsRepo":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # SQLAlchemy engine handles connection cleanup automatically
        pass

    def query(self, query: LiteralString, params: Union[Tuple[Any, ...], Dict[str, Any]] = ()) -> Sequence[Row]:
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return result.fetchall()

    def execute(self, query: LiteralString, params: Union[Tuple[Any, ...], Dict[str, Any]] = ()) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(query), params)

    def get_policy_ids(self, policy_names: List[str]) -> Dict[str, int]:
        if not policy_names:
            return {}

        with self.engine.connect() as conn:
            # Use SQLAlchemy's parameter binding
            result = conn.execute(
                text("SELECT id, name FROM policies WHERE name = ANY(:policy_names)"),
                {"policy_names": policy_names},
            )
            return {row[1]: row[0] for row in result.fetchall()}

    def create_training_run(self, name: str, user_id: str, attributes: Dict[str, str], url: str | None) -> int:
        status = "running"
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO training_runs (name, user_id, attributes, status, url)
                    VALUES (:name, :user_id, :attributes, :status, :url)
                    RETURNING id
                """),
                {
                    "name": name,
                    "user_id": user_id,
                    "attributes": attributes,
                    "status": status,
                    "url": url,
                },
            )
            row = result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert training run")
            return row[0]

    def create_policy_epoch(
        self, run_id: int, start_training_epoch: int, end_training_epoch: int, attributes: Dict[str, str]
    ) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO policy_epochs (run_id, start_training_epoch, end_training_epoch, attributes)
                    VALUES (:run_id, :start_training_epoch, :end_training_epoch, :attributes)
                    RETURNING id
                """),
                {
                    "run_id": run_id,
                    "start_training_epoch": start_training_epoch,
                    "end_training_epoch": end_training_epoch,
                    "attributes": attributes,
                },
            )
            row = result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert policy epoch")
            return row[0]

    def create_policy(self, name: str, description: str | None, url: str | None, epoch_id: int | None) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO policies (name, description, url, epoch_id)
                    VALUES (:name, :description, :url, :epoch_id)
                    RETURNING id
                """),
                {
                    "name": name,
                    "description": description,
                    "url": url,
                    "epoch_id": epoch_id,
                },
            )
            row = result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert policy")
            return row[0]

    def record_episode(
        self,
        agent_policies: Dict[int, int],
        agent_metrics: Dict[int, Dict[str, float]],
        eval_name: str | None,
        simulation_suite: str | None,
        replay_url: str | None,
        attributes: Dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            # Insert into episodes table
            result = conn.execute(
                text("""
                    INSERT INTO episodes (
                        replay_url,
                        eval_name,
                        simulation_suite,
                        attributes
                    ) VALUES (
                        :replay_url, :eval_name, :simulation_suite, :attributes
                    ) RETURNING id
                """),
                {
                    "replay_url": replay_url,
                    "eval_name": eval_name,
                    "simulation_suite": simulation_suite,
                    "attributes": attributes,
                },
            )
            row = result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert episode record")
            episode_id = row[0]

            # Insert agent policies
            for agent_id, policy_id in agent_policies.items():
                conn.execute(
                    text("""
                        INSERT INTO episode_agent_policies (
                            episode_id,
                            policy_id,
                            agent_id
                        ) VALUES (:episode_id, :policy_id, :agent_id)
                    """),
                    {
                        "episode_id": episode_id,
                        "policy_id": policy_id,
                        "agent_id": agent_id,
                    },
                )

            # Insert agent metrics
            for agent_id, metrics in agent_metrics.items():
                for metric_name, value in metrics.items():
                    conn.execute(
                        text("""
                            INSERT INTO episode_agent_metrics (
                                episode_id,
                                agent_id,
                                metric,
                                value
                            ) VALUES (:episode_id, :agent_id, :metric, :value)
                        """),
                        {
                            "episode_id": episode_id,
                            "agent_id": agent_id,
                            "metric": metric_name,
                            "value": value,
                        },
                    )
