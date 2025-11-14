from metta.app_backend.schema_manager import SqlMigration

# This is a list of migrations that will be applied to the eval database.
# Do not change existing migrations, only add new ones.
MIGRATIONS = [
    SqlMigration(
        version=0,
        description="Initial schema",
        sql_statements=[
            """CREATE EXTENSION IF NOT EXISTS "uuid-ossp" """,
            """CREATE TABLE policies (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                attributes JSONB,
                UNIQUE (user_id, name)
            )""",
            """CREATE TABLE policy_versions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                internal_id SERIAL UNIQUE,
                policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                s3_path TEXT,
                git_hash TEXT,
                policy_spec JSONB,
                attributes JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (policy_id, version)
            )""",
            """CREATE TABLE episodes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                internal_id SERIAL UNIQUE,
                data_uri TEXT NOT NULL,
                primary_pv_id UUID REFERENCES policy_versions(id) ON DELETE CASCADE,
                replay_url TEXT,
                thumbnail_url TEXT,
                attributes JSONB,
                eval_task_id UUID,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE episode_tags (
                episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (episode_id, key)
            )""",
            """CREATE TABLE episode_policies (
                episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                policy_version_id UUID NOT NULL REFERENCES policy_versions(id) ON DELETE CASCADE,
                num_agents INTEGER NOT NULL,
                PRIMARY KEY (episode_id, policy_version_id)
            )""",
            """CREATE TABLE episode_policy_metrics (
              episode_internal_id INTEGER NOT NULL REFERENCES episodes(internal_id) ON DELETE CASCADE,
              pv_internal_id INTEGER NOT NULL REFERENCES policy_versions(internal_id) ON DELETE CASCADE,
              metric_name TEXT NOT NULL,
              value FLOAT NOT NULL,
              PRIMARY KEY (episode_internal_id, pv_internal_id, metric_name)
            )""",
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
            """CREATE TABLE eval_tasks (
                id SERIAL UNIQUE PRIMARY KEY,
                command TEXT NOT NULL,
                git_hash TEXT,
                attributes JSONB,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                status TEXT NOT NULL DEFAULT 'unprocessed',
                status_details JSONB,
                assigned_at TIMESTAMP,
                assignee TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                output_log_path TEXT
            )""",
        ],
    ),
]
