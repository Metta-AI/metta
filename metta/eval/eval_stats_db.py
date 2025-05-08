
    # --------------------------------------------------------------------------- #
    #  policy simulations view                                                    #
    # --------------------------------------------------------------------------- #

    def materialize_policy_simulations_view(self, metric: str = "reward") -> None:
        logger = logging.getLogger(__name__)
        logger.debug(f"Creating materialized view for metric '{metric}'")

        table_name = f"policy_simulations_{metric}"

        if not self._metric_exists(metric):
            raise ValueError(f"Metric '{metric}' not found in the agent_metrics table")

        try:
            self.con.execute("BEGIN TRANSACTION")

            # First, create the table structure with primary key
            self._create_policy_simulations_table_structure(metric, table_name)
            logger.debug(f"Created table structure for {table_name}")

            # Then populate it with data
            self._populate_policy_simulations_table(metric, table_name)

            # Log what got inserted
            row_count = self.con.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()[0]
            logger.debug(f"Inserted {row_count} rows into {table_name}")

            # Add index on simulation fields
            self.con.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_sim 
            ON {table_name}(sim_suite, sim_name)
            """)

            self.con.execute("COMMIT")
            logger.info(f"Successfully created materialized view '{table_name}' for metric '{metric}'")
        except Exception as e:
            self.con.execute("ROLLBACK")
            logger.error(f"Failed to create materialized view: {e}")
            raise

    # --------------------------------------------------------------------------- #
    # internal helpers                                                            #
    # --------------------------------------------------------------------------- #
    def _metric_exists(self: StatsDB, metric: str) -> bool:
        query = f"SELECT COUNT(*) FROM agent_metrics WHERE metric = '{metric}'"
        try:
            result = self.con.execute(query).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def _create_policy_simulations_table_structure(self: StatsDB, metric: str, table_name: str) -> None:
        # Drop table if it exists
        self.con.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create table with modified primary key that handles NULL policy_version
        self.con.execute(f"""
        CREATE TABLE {table_name} (
            policy_key      TEXT NOT NULL,
            policy_version  INT NOT NULL,
            sim_suite       TEXT NOT NULL,
            sim_name        TEXT NOT NULL,
            sim_env         TEXT NOT NULL,
            {metric}        DOUBLE,
            {metric}_std    DOUBLE,
            PRIMARY KEY (policy_key, policy_version, sim_suite, sim_env)
        )
        """)

    def _populate_policy_simulations_table(self: StatsDB, metric: str, table_name: str) -> None:
        """Populate the policy simulations table with aggregated metrics."""
        logger = logging.getLogger(__name__)

        # Check if we have any episodes
        episode_count = self.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        if episode_count == 0:
            logger.warning("No episodes found in database!")
            return

        sql = f"""
        WITH with_ctx AS (
            SELECT
                ap.policy_key,
                ap.policy_version,
                s.suite AS sim_suite,
                s.name AS sim_name,
                s.env AS sim_env,
                am.value AS metric_value
            FROM agent_metrics am
            JOIN agent_policies ap
                ON ap.episode_id = am.episode_id
                AND ap.agent_id = am.agent_id
            JOIN episodes e ON e.id = am.episode_id
            JOIN simulations s ON s.id = e.simulation_id
            WHERE am.metric = '{metric}'
        )
        INSERT INTO {table_name}
        SELECT 
            policy_key,
            policy_version,
            sim_suite,
            sim_name,
            sim_env,
            AVG(metric_value) AS {metric},
            STDDEV_SAMP(metric_value) AS {metric}_std
        FROM with_ctx
        GROUP BY
            policy_key, policy_version, sim_suite, sim_name, sim_env
        """

        try:
            self.con.execute(sql)
            # Log the count of rows inserted
            count_result = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            logger.info(f"Inserted {count_result[0]} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error executing populate query: {e}")
            raise

    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        view_name = f"policy_simulations_{metric}"

        # Build the query
        query = f"""
            SELECT AVG({view_name}.{metric}) as score
            FROM {view_name}
            WHERE {view_name}.policy_key = '{policy_key}'
            AND {view_name}.policy_version = {policy_version}
            """

        # Add optional filter condition
        if filter_condition:
            query += f" AND {filter_condition}"

        # Execute the query
        result = self.query(query)
        if not result.empty and not pd.isna(result["score"][0]):
            return float(result["score"][0])
        return None

    def simulation_scores(self, policy_key: str, policy_version: int, metric: str) -> dict:
        """
        Get all simulation scores for a specific policy and metric.

        Args:
            policy_key (str): The policy key
            policy_version (int): The policy version
            metric (str): The metric name

        Returns:
            dict: A dictionary mapping (sim_suite, sim_name, sim_env) tuples to metric values
        """

        # Ensure view exists
        view_name = f"policy_simulations_{metric}"
        query = f"""
        SELECT sim_suite, sim_name, sim_env, {metric}
        FROM {view_name}
        WHERE policy_key = '{policy_key}' AND policy_version = {policy_version}
        """

        results = self.query(query)

        # Return empty dict if no results
        if results.empty:
            return {}

        # Create mapping from (sim_suite, sim_name, sim_env) -> metric value
        scores = {}
        for _, row in results.iterrows():
            key = (row["sim_suite"], row["sim_name"], row["sim_env"])
            scores[key] = row[metric]

        return scores
