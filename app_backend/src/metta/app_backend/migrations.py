from metta.app_backend.schema_manager import SqlMigration

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
    SqlMigration(
        version=27,
        description="Add cogames_policy_submissions table",
        sql_statements=[
            """CREATE TABLE cogames_policy_submissions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id TEXT NOT NULL,
                name TEXT,
                s3_path TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE INDEX idx_cogames_submissions_user_id ON cogames_policy_submissions(user_id)""",
            """CREATE INDEX idx_cogames_submissions_created_at ON cogames_policy_submissions(created_at)""",
        ],
    ),
    SqlMigration(
        version=28,
        description="Drop leaderboard-related tables",
        sql_statements=[
            """DROP TABLE IF EXISTS leaderboard_policy_scores""",
            """DROP TABLE IF EXISTS leaderboards""",
        ],
    ),
    SqlMigration(
        version=29,
        description="Drop more tables",
        sql_statements=[
            """DROP TABLE IF EXISTS saved_dashboards""",
            """DROP VIEW IF EXISTS wide_episodes""",
            """DROP VIEW IF EXISTS unified_training_runs""",
        ],
    ),
]
