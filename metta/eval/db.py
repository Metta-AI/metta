"""
Database interface for storing and querying policy evaluation metrics.
"""

import logging
import os
import sqlite3
from typing import Dict, List, Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig

from metta.rl.eval.eval_stats_db import EvalStatsDB
from metta.rl.wandb.wandb_context import WandbContext

logger = logging.getLogger(__name__)


class PolicyEvalDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS policies (
        uri TEXT NOT NULL,       /* The policy uri */
        version TEXT NOT NULL,    /* The parsed version */
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (uri, version)
    );

    CREATE TABLE IF NOT EXISTS evaluations (
        name TEXT NOT NULL,
        metric TEXT NOT NULL,
        PRIMARY KEY (name, metric)
    );

    CREATE TABLE IF NOT EXISTS policy_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_uri TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        evaluation_name TEXT NOT NULL,
        metric TEXT NOT NULL,
        mean REAL,
        stdev REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (policy_uri, policy_version, evaluation_name, metric),
        FOREIGN KEY (policy_uri, policy_version) REFERENCES policies(uri, version),
        FOREIGN KEY (evaluation_name, metric) REFERENCES evaluations(name, metric)
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = self._create_connection(db_path)
        self._init_schema()

    ##
    ## Database Connections
    ##

    def _create_connection(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _begin(self):
        self.conn.execute("BEGIN TRANSACTION")

    def _commit(self):
        self.conn.commit()

    def _rollback(self):
        self.conn.rollback()

    def _init_schema(self):
        cursor = self.conn.cursor()

        try:
            self._begin()
            cursor.execute("PRAGMA foreign_keys = ON")
            statements = [s.strip() for s in self.SCHEMA.split(";") if s.strip()]
            for statement in statements:
                cursor.execute(statement)
            self._commit()
            logger.info(f"Schema initialized successfully for {self.db_path}")

        except Exception as e:
            self._rollback()
            logger.error(f"Error initializing schema: {e}")
            raise
        finally:
            cursor.close()

    def _execute(self, sql: str, params: Tuple = None):
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql, params or ())
            return cursor
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            logger.error(f"Query: {sql}")
            logger.error(f"Params: {params}")
            raise

    def _executemany(self, sql: str, params_list: List[Tuple]):
        cursor = self.conn.cursor()
        try:
            cursor.executemany(sql, params_list)
            return cursor
        except Exception as e:
            logger.error(f"SQL executemany error: {e}")
            logger.error(f"Query: {sql}")
            raise

    def query(self, sql: str, params: Tuple = None) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def parse_versioned_uri(self, uri: str) -> Tuple[str, str]:
        """
        Parse a versioned policy uri into uri and version.
        Examples:
        - "navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
        - "wandb://run/navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
        """
        # Remove any prefix
        if "://" in uri:
            uri = uri.split("/")[-1]

        # Extract version if present
        if ":v" in uri:
            parts = uri.split(":v")
            name = parts[0]
            version = "v" + parts[1]
        else:
            # If no version specified, treat the entire URI as the name and set version to "latest"
            name = uri
            version = "latest"

        return name, version

    ##
    ## Data Loading
    ##

    def _construct_metric_to_df_map(self, cfg: DictConfig, dfs: list) -> Dict[str, pd.DataFrame]:
        metrics = [m.metric for m in cfg.analyzer.analysis.metrics]

        if len(metrics) != len(dfs):
            raise ValueError(f"Mismatch between metrics ({len(metrics)}) and dataframes ({len(dfs)})")

        metric_to_df = {}
        for metric, df in zip(metrics, dfs, strict=False):
            metric_to_df[metric] = df

        return metric_to_df

    def short_name(self, name: str) -> str:
        return name.split("/")[-1]

    def import_from_eval_stats(self, cfg: DictConfig):
        """
        Expected dataframe schema for each metric:
        - policy_uri: String identifier for the policy
        - eval_name: String identifier for the evaluation environment
        - mean_{metric}: Float value representing the mean of the metric for this policy in this eval
        - std_{metric}: Float value representing the standard deviation of the metric
        """
        logger.info(f"Importing data from {cfg.eval.eval_db_uri}")
        with WandbContext(cfg) as wandb_run:
            eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)

        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        dfs, _ = analyzer.analyze(include_policy_fitness=False)
        metric_to_df = self._construct_metric_to_df_map(cfg, dfs)

        # Track policies and evaluation metrics we've already created
        created_policies = set()
        created_metrics = set()

        # A list to collect all the evaluation results for batch insertion
        results_to_insert = []

        # Start transaction for the entire import process
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for metric, df in metric_to_df.items():
                if df is None or df.empty:
                    logger.warning(f"No data found for metric {metric}")
                    continue

                logger.info(f"Processing {len(df)} results for metric {metric}")

                # Process each policy-environment pair
                for _, row in df.iterrows():
                    policy_uri = row["policy_name"]
                    uri, version = self.parse_versioned_uri(policy_uri)
                    evaluation_name = row["eval_name"]

                    # Create policy if needed
                    policy_key = (uri, version)
                    if policy_key not in created_policies:
                        cursor = self.conn.execute(
                            "SELECT 1 FROM policies WHERE uri = ? AND version = ?", (uri, version)
                        )
                        if not cursor.fetchone():
                            self.conn.execute("INSERT INTO policies (uri, version) VALUES (?, ?)", (uri, version))
                        created_policies.add(policy_key)

                    # Create metric if needed
                    metric_key = (evaluation_name, metric)
                    if metric_key not in created_metrics:
                        self._execute("INSERT INTO evaluations (name, metric) VALUES (?, ?)", (evaluation_name, metric))
                        created_metrics.add(metric_key)

                    # Extract metric values - find the appropriate column names
                    mean_col = [col for col in row.index if col.startswith("mean_")][0]
                    std_col = [col for col in row.index if col.startswith("std_")][0]

                    # Add to batch list
                    results_to_insert.append((uri, version, evaluation_name, metric, row[mean_col], row[std_col]))

            # Insert all results in a batch
            if results_to_insert:
                # Insert with conflict handling since unique constraint exists
                logger.info(f"Batch inserting {len(results_to_insert)} evaluation results")
                self.conn.executemany(
                    """INSERT OR IGNORE INTO policy_evaluations 
                    (policy_uri, policy_version, evaluation_name, metric, mean, stdev) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    results_to_insert,
                )

            # Commit the transaction
            # Commit the transaction
            self.conn.commit()
            logger.info(
                f"Import completed with {len(created_policies)} policies, "
                f"{len(created_metrics)} metrics, "
                f"{len(results_to_insert)} results"
            )

        except Exception as e:
            # Rollback on error
            logger.error(f"Import failed, transaction rolled back: {e}")
            raise

    def get_matrix_data(self, metric: str, view_type: str = "latest", policy_uri: str = None) -> pd.DataFrame:
        """
        Get matrix data for the specified metric.

        Args:
            metric: The metric to get data for
            view_type:
                - "latest": Only the latest version of each policy, sorted by score
                - "all": All versions of all policies, sorted by score
                - "policy_versions": All versions of a specific policy, sorted by version
                - "chronological": All policies and versions, sorted by creation date
            policy_uri: Required for "policy_versions" view_type

        Returns:
            DataFrame with policies as rows and evaluations as columns
        """
        if policy_uri:
            policy_uri = self.parse_versioned_uri(policy_uri)[0]

        # Execute the appropriate query based on view type
        if view_type == "latest":
            # Get the latest version for each policy
            sql = """
            WITH latest_versions AS (
                SELECT uri, MAX(version) as version
                FROM policies
                GROUP BY uri
            )
            SELECT 
                p.uri || ':' || p.version as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.policy_uri = p.uri AND pe.policy_version = p.version
            JOIN latest_versions lv ON p.uri = lv.uri AND p.version = lv.version
            WHERE pe.metric = ?
            """
            df = self.query(sql, (metric,))
        elif view_type == "policy_versions" and policy_uri:
            # Get all versions of a specific policy
            sql = """
            SELECT 
                p.uri || ':' || p.version as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.policy_uri = p.uri AND pe.policy_version = p.version
            WHERE pe.metric = ? AND p.uri = ?
            """
            df = self.query(sql, (metric, policy_uri))
        elif view_type == "chronological":
            # All policies and versions, sorted by creation date
            sql = """
            SELECT 
                p.uri || ':' || p.version as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.policy_uri = p.uri AND pe.policy_version = p.version
            WHERE pe.metric = ?
            ORDER BY p.created_at ASC
            """
            df = self.query(sql, (metric,))
        else:  # "all" or default
            # Get all versions of all policies
            sql = """
            SELECT 
                p.uri || ':' || p.version as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.policy_uri = p.uri AND pe.policy_version = p.version
            WHERE pe.metric = ?
            """
            df = self.query(sql, (metric,))

        if len(df) == 0:
            logger.warning(f"No data found for metric {metric}")
            return pd.DataFrame()

        # Process data into matrix format
        df["display_name"] = df["evaluation_name"].apply(self.short_name)

        policies = df["policy_uri"].unique()
        eval_display_names = df["display_name"].unique()

        # Generate an overall score for each policy
        overall_scores = {}
        for policy in policies:
            policy_data = df[df["policy_uri"] == policy]
            overall_scores[policy] = policy_data["value"].mean()

        # Create mapping for easy lookup
        data_map = {}
        for _, row in df.iterrows():
            data_map[(row["policy_uri"], row["display_name"])] = row["value"]

        # Create matrix data
        matrix_data = []
        for policy in policies:
            row_data = {"policy_uri": policy}
            row_data["Overall"] = overall_scores[policy]
            for display_name in eval_display_names:
                key = (policy, display_name)
                if key in data_map:
                    row_data[display_name] = data_map[key]
            matrix_data.append(row_data)

        # Convert to DataFrame
        matrix = pd.DataFrame(matrix_data)
        matrix = matrix.set_index("policy_uri")

        # Sort appropriately based on view_type
        if view_type in ["latest", "all"]:
            # Sort by overall score (lowest first)
            sorted_policies = sorted(policies, key=lambda p: overall_scores[p])
            matrix = matrix.reindex(sorted_policies)
        elif view_type == "policy_versions":
            # For policy versions view, sort by version number for clear progression
            if policy_uri:
                # Extract version numbers and convert to sortable format
                def version_key(uri):
                    parts = uri.split(":v")
                    if len(parts) > 1 and parts[1].isdigit():
                        return int(parts[1])
                    return uri  # Fallback to string comparison

                sorted_policies = sorted(policies, key=version_key)
                matrix = matrix.reindex(sorted_policies)
        # For chronological view, we don't need to re-sort as it's already sorted in the SQL query

        return matrix
