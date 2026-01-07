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
                data_uri TEXT,
                git_hash TEXT,
                attributes JSONB,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_finished BOOLEAN NOT NULL DEFAULT FALSE,
                latest_attempt_id INTEGER
            )""",
            """CREATE TABLE task_attempts (
                id SERIAL UNIQUE PRIMARY KEY,
                task_id INTEGER NOT NULL REFERENCES eval_tasks(id) ON DELETE CASCADE,
                attempt_number INTEGER NOT NULL DEFAULT 0,
                assigned_at TIMESTAMP,
                assignee TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                output_log_path TEXT,
                status TEXT NOT NULL DEFAULT 'unprocessed',
                status_details JSONB
            )""",
            """ALTER TABLE eval_tasks
                ADD CONSTRAINT eval_tasks_latest_attempt_fkey
                FOREIGN KEY (latest_attempt_id) REFERENCES task_attempts(id) ON DELETE CASCADE
            """,
            """CREATE VIEW eval_tasks_view AS
                SELECT
                    t.id, t.command, t.data_uri, t.git_hash, t.attributes,
                    t.user_id, t.created_at, t.is_finished, t.latest_attempt_id,
                    a.attempt_number, a.status, a.status_details, a.assigned_at,
                    a.assignee, a.started_at, a.finished_at, a.output_log_path
                FROM eval_tasks t
                LEFT JOIN task_attempts a ON t.latest_attempt_id = a.id
            """,
        ],
    ),
    SqlMigration(
        version=1,
        description="Create policy_version_tags table for tag-based queries",
        sql_statements=[
            """CREATE TABLE policy_version_tags (
                policy_version_id UUID NOT NULL REFERENCES policy_versions(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (policy_version_id, key)
            )""",
            """CREATE INDEX idx_policy_version_tags_key_value
               ON policy_version_tags (key, value)""",
        ],
    ),
    SqlMigration(
        version=2,
        description="Add indexes on episode_tags for key-value queries",
        sql_statements=[
            """CREATE INDEX idx_episode_tags_key_value
               ON episode_tags (key, value)""",
            """CREATE INDEX idx_episode_tags_episode_key_value
               ON episode_tags (episode_id, key, value)""",
        ],
    ),
    SqlMigration(
        version=3,
        description="Create job_requests table for job orchestration",
        sql_statements=[
            """CREATE TYPE job_status AS ENUM ('pending', 'dispatched', 'running', 'completed', 'failed')""",
            """CREATE TABLE job_requests (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                job_type TEXT NOT NULL,
                job JSONB NOT NULL,
                user_id TEXT NOT NULL,
                status job_status NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                dispatched_at TIMESTAMP,
                running_at TIMESTAMP,
                completed_at TIMESTAMP,
                worker TEXT,
                result JSONB,
                error TEXT
            )""",
            """CREATE INDEX idx_job_requests_type_status_created ON job_requests (job_type, status, created_at DESC)""",
            """CREATE INDEX idx_job_requests_type_created ON job_requests (job_type, created_at DESC)""",
            """CREATE INDEX idx_job_requests_created ON job_requests (created_at DESC)""",
            """CREATE INDEX idx_job_requests_status ON job_requests (status)""",
            """CREATE INDEX idx_job_requests_user_id ON job_requests (user_id)""",
        ],
    ),
    SqlMigration(
        version=4,
        description="Create tournament tables",
        sql_statements=[
            """CREATE TYPE match_status AS ENUM ('pending', 'scheduled', 'running', 'completed', 'failed')""",
            """CREATE TYPE membership_action AS ENUM ('add', 'remove')""",
            """CREATE TABLE seasons (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_seasons_name ON seasons (name)""",
            """CREATE TABLE pools (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                season_id UUID REFERENCES seasons(id) ON DELETE CASCADE,
                name TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_pools_season_id ON pools (season_id)""",
            """CREATE TABLE pool_players (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                pool_id UUID NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                policy_version_id UUID NOT NULL REFERENCES policy_versions(id) ON DELETE CASCADE,
                retired BOOLEAN NOT NULL DEFAULT FALSE,
                UNIQUE (pool_id, policy_version_id)
            )""",
            """CREATE INDEX idx_pool_players_pool_id ON pool_players (pool_id)""",
            """CREATE INDEX idx_pool_players_policy_version_id ON pool_players (policy_version_id)""",
            """CREATE TABLE matches (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                pool_id UUID NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
                job_id UUID REFERENCES job_requests(id) ON DELETE SET NULL,
                assignments INTEGER[] NOT NULL,
                status match_status NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )""",
            """CREATE INDEX idx_matches_pool_id ON matches (pool_id)""",
            """CREATE INDEX idx_matches_job_id ON matches (job_id)""",
            """CREATE INDEX idx_matches_status ON matches (status)""",
            """CREATE TABLE match_players (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                pool_player_id UUID NOT NULL REFERENCES pool_players(id) ON DELETE CASCADE,
                policy_index INTEGER NOT NULL DEFAULT 0,
                score FLOAT
            )""",
            """CREATE INDEX idx_match_players_match_id ON match_players (match_id)""",
            """CREATE INDEX idx_match_players_pool_player_id ON match_players (pool_player_id)""",
            """CREATE TABLE membership_changes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                pool_player_id UUID NOT NULL REFERENCES pool_players(id) ON DELETE CASCADE,
                action membership_action NOT NULL,
                notes TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_membership_changes_pool_player_id ON membership_changes (pool_player_id)""",
            """CREATE INDEX idx_membership_changes_created_at ON membership_changes (created_at DESC)""",
        ],
    ),
    SqlMigration(
        version=5,
        description="Add created_at to pool_players",
        sql_statements=[
            """ALTER TABLE pool_players ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP""",
        ],
    ),
]
