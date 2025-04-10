"""
Database interface for storing and querying policy evaluation metrics.
"""

import os
import sqlite3
import logging
import pandas as pd
from omegaconf import DictConfig

from typing import Dict, Tuple, List, Optional
import hydra
from rl.wandb.wandb_context import WandbContext
from rl.eval.eval_stats_db import EvalStatsDB
from util.s3_utils import download_from_s3, upload_to_s3

logger = logging.getLogger(__name__)

class PolicyEvalDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS policies (
        uri TEXT PRIMARY KEY,     /* The full policy URI */
        name TEXT NOT NULL,       /* The parsed policy name without version */
        version TEXT NOT NULL,    /* The parsed version */
        complete BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS evaluations (
        name TEXT NOT NULL,
        metric TEXT NOT NULL,
        PRIMARY KEY (name, metric)
    );

    CREATE TABLE IF NOT EXISTS policy_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uri TEXT NOT NULL,        /* Foreign key to policies.uri */
        evaluation_name TEXT NOT NULL,
        metric TEXT NOT NULL,
        mean REAL,
        stdev REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (uri, evaluation_name, metric),
        FOREIGN KEY (uri) REFERENCES policies(uri),
        FOREIGN KEY (evaluation_name, metric) REFERENCES evaluations(name, metric)
    );

    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uri TEXT NOT NULL,                /* Foreign key to policies.uri */
        evaluation_name TEXT NOT NULL,    /* The evaluation environment */
        metric TEXT NOT NULL,             /* The metric to evaluate */
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        FOREIGN KEY (uri) REFERENCES policies(uri),
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
            statements = [s.strip() for s in self.SCHEMA.split(';') if s.strip()]
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

    @staticmethod
    def from_uri(uri: str, run_dir: str = None, wandb_run = None):
        """
        Create a PolicyEvalDB from a URI, which can be a local path or S3 path.
        """
        if uri.startswith("s3://"):
            # Download from S3 to local path
            local_path = os.path.join(run_dir, "policy_eval.db")
            download_from_s3(uri, local_path)
            return PolicyEvalDB(local_path)
        else:
            if uri.startswith("file://"):
                db_path = uri.split("file://")[1]
            else:
                db_path = uri
            return PolicyEvalDB(db_path)
    
    def upload_to_s3(self, s3_path: str):
        """Upload the database to S3."""
        return upload_to_s3(self.db_path, s3_path)

    def parse_policy_uri(self, uri: str) -> Tuple[str, str]:
        """
        Parse a policy URI into name and version.
        Examples:
        - "navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
        - "wandb://run/navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
        """
        # Remove any prefix
        if "://" in uri:
            uri = uri.split("/")[-1]
        
        # Extract version if present
        if ':v' in uri:
            parts = uri.split(':v')
            name = parts[0]
            version = 'v' + parts[1]
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
        for metric, df in zip(metrics, dfs):
            metric_to_df[metric] = df
        
        return metric_to_df
    
    def short_name(self, name: str) -> str:
        return name.split('/')[-1]

    def import_from_eval_stats(self, cfg: DictConfig):
        """
        Expected dataframe schema for each metric:
        - policy_name: String identifier for the policy
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
                    policy_uri = row['policy_name']
                    name, version = self.parse_policy_uri(policy_uri)
                    evaluation_name = row['eval_name']
                    
                    # Create policy if needed
                    if policy_uri not in created_policies:   
                        cursor = self.conn.execute(
                            "SELECT 1 FROM policies WHERE uri = ?", 
                            (policy_uri,)
                        )
                        if not cursor.fetchone():
                            self.conn.execute(
                                "INSERT INTO policies (uri, name, version) VALUES (?, ?, ?)",
                                (policy_uri, name, version)
                            )
                        created_policies.add(policy_uri)
                    
                    # Create metric if needed
                    metric_key = (evaluation_name, metric)
                    if metric_key not in created_metrics:
                        self._execute(
                            "INSERT INTO evaluations (name, metric) VALUES (?, ?)",
                            (evaluation_name, metric)
                        )
                        created_metrics.add(metric_key)
                    
                    # Extract metric values - find the appropriate column names
                    mean_col = [col for col in row.index if col.startswith('mean_')][0]
                    std_col = [col for col in row.index if col.startswith('std_')][0]
                    
                    # Add to batch list
                    results_to_insert.append((
                        policy_uri,
                        evaluation_name,
                        metric,
                        row[mean_col],
                        row[std_col]
                    ))
            
            # Insert all results in a batch
            if results_to_insert:
                # Insert with conflict handling since unique constraint exists
                logger.info(f"Batch inserting {len(results_to_insert)} evaluation results")
                self.conn.executemany(
                    """INSERT OR IGNORE INTO policy_evaluations 
                    (uri, evaluation_name, metric, mean, stdev) 
                    VALUES (?, ?, ?, ?, ?)""",
                    results_to_insert
                )
            
            # Commit the transaction
            self.conn.commit()
            logger.info(f"Import completed with {len(created_policies)} policies, {len(created_metrics)} metrics, {len(results_to_insert)} results")
            
        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Import failed, transaction rolled back: {e}")
            raise

    def get_matrix_data(self, metric: str, view_type: str = "latest", policy_name: str = None) -> pd.DataFrame:
        """
        Get matrix data for the specified metric.
        
        Args:
            metric: The metric to get data for
            view_type: 
                - "latest": Only the latest version of each policy, sorted by score
                - "all": All versions of all policies, sorted by score
                - "policy_versions": All versions of a specific policy, sorted by version
                - "chronological": All policies and versions, sorted by creation date
            policy_name: Required for "policy_versions" view_type
            
        Returns:
            DataFrame with policies as rows and evaluations as columns
        """
        if view_type == "latest":
            # Get the latest version for each policy
            sql = """
            WITH latest_versions AS (
                SELECT name, MAX(version) as version
                FROM policies
                GROUP BY name
            )
            SELECT 
                p.uri as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.uri = p.uri
            JOIN latest_versions lv ON p.name = lv.name AND p.version = lv.version
            WHERE pe.metric = ?
            """
        elif view_type == "policy_versions" and policy_name:
            # Get all versions of a specific policy
            sql = """
            SELECT 
                p.uri as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.uri = p.uri
            WHERE pe.metric = ? AND p.name = ?
            ORDER BY p.version ASC
            """
            return self.query(sql, (metric, policy_name))
        elif view_type == "chronological":
            # All policies and versions, sorted by creation date
            sql = """
            SELECT 
                p.uri as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.uri = p.uri
            WHERE pe.metric = ?
            ORDER BY p.created_at ASC
            """
        else:  # "all" or default
            # Get all versions of all policies
            sql = """
            SELECT 
                p.uri as policy_uri,
                pe.evaluation_name, 
                pe.mean as value
            FROM policy_evaluations pe
            JOIN policies p ON pe.uri = p.uri
            WHERE pe.metric = ?
            """
        
        df = self.query(sql, (metric,))
        if len(df) == 0:
            logger.warning(f"No data found for metric {metric}")
            return pd.DataFrame()
        
        # Process data into matrix format
        df['display_name'] = df['evaluation_name'].apply(self.short_name)
        
        policies = df['policy_uri'].unique()
        eval_display_names = df['display_name'].unique()
        
        # Generate an overall score for each policy
        overall_scores = {}
        for policy in policies:
            policy_data = df[df['policy_uri'] == policy]
            overall_scores[policy] = policy_data['value'].mean()
        
        # Create mapping for easy lookup
        data_map = {}
        for _, row in df.iterrows():
            data_map[(row['policy_uri'], row['display_name'])] = row['value']
        
        # Create matrix data
        matrix_data = []
        for policy in policies:
            row_data = {'policy_uri': policy}
            row_data['Overall'] = overall_scores[policy]
            for display_name in eval_display_names:
                key = (policy, display_name)
                if key in data_map:
                    row_data[display_name] = data_map[key]
            matrix_data.append(row_data)
        
        # Convert to DataFrame
        matrix = pd.DataFrame(matrix_data)
        matrix = matrix.set_index('policy_uri')
        
        # Sort appropriately
        if view_type in ["latest", "all"]:
            # Sort by overall score
            policy_order = sorted(policies, key=lambda p: overall_scores[p], reverse=True)
            matrix = matrix.reindex(policy_order)
        elif view_type == "policy_versions":
            # Already sorted by version in SQL
            pass
        elif view_type == "chronological":
            # Already sorted by creation date in SQL
            pass
        
        return matrix

    ##
    ## Policy Management
    ##

    def add_policies(self, policies: List[Tuple[str, str, str, bool]]):
        """
        Add a list of policies to the database.
        
        Args:
            policies: List of tuples (uri, name, version, complete)
        """
        if not policies:
            return
            
        try:
            self.conn.execute("BEGIN TRANSACTION")
            
            # Add to policies table
            for uri, name, version, is_complete in policies:
                self.conn.execute(
                    "INSERT OR IGNORE INTO policies (uri, name, version, complete) VALUES (?, ?, ?, ?)",
                    (uri, name, version, 1 if is_complete else 0)
                )
            
            self.conn.commit()
            logger.info(f"Added {len(policies)} policies to the database")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding policies: {e}")
            raise

    def get_policies(self):
        """
        Get all policies from the database.
        
        Returns:
            DataFrame with policy details
        """
        return self.query("""
            SELECT uri, name, version, complete, created_at
            FROM policies
            ORDER BY name, version
        """)

    def mark_policy_complete(self, uri: str, is_complete: bool = True):
        """
        Mark a policy as complete (training finished).
        
        Args:
            uri: The policy URI
            is_complete: Whether the policy is complete
        """
        self.conn.execute(
            "UPDATE policies SET complete = ? WHERE uri = ?",
            (1 if is_complete else 0, uri)
        )
        self.conn.commit()

    ##
    ## Job Queue Management
    ##

    def add_job(self, uri: str, evaluation_name: str, metric: str):
        """
        Add a policy evaluation job to the queue.
        
        Args:
            uri: The policy URI
            evaluation_name: The evaluation to run
            metric: The metric to evaluate
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")
            # Check if already in queue
            cursor = self.conn.execute(
                "SELECT 1 FROM jobs WHERE uri = ? AND evaluation_name = ? AND metric = ?",
                (uri, evaluation_name, metric)
            )
            if not cursor.fetchone():
                self.conn.execute(
                    "INSERT INTO jobs (uri, evaluation_name, metric) VALUES (?, ?, ?)",
                    (uri, evaluation_name, metric)
                )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding job: {e}")
            raise

    def add_jobs_for_policy(self, uri: str, evaluations: List[Tuple[str, str]]):
        """
        Add multiple jobs for a policy to the queue.
        
        Args:
            uri: The policy URI
            evaluations: List of (evaluation_name, metric) tuples
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")
            
            # Check if policy exists
            cursor = self.conn.execute("SELECT 1 FROM policies WHERE uri = ?", (uri,))
            if not cursor.fetchone():
                raise ValueError(f"Policy not found: {uri}")
            
            # Add each evaluation to jobs queue
            for evaluation_name, metric in evaluations:
                # Check if evaluation exists
                cursor = self.conn.execute(
                    "SELECT 1 FROM evaluations WHERE name = ? AND metric = ?", 
                    (evaluation_name, metric)
                )
                if not cursor.fetchone():
                    # Create evaluation if it doesn't exist
                    self.conn.execute(
                        "INSERT INTO evaluations (name, metric) VALUES (?, ?)",
                        (evaluation_name, metric)
                    )
                
                # Check if job already exists
                cursor = self.conn.execute(
                    "SELECT 1 FROM jobs WHERE uri = ? AND evaluation_name = ? AND metric = ?",
                    (uri, evaluation_name, metric)
                )
                if not cursor.fetchone():
                    # Add job
                    self.conn.execute(
                        "INSERT INTO jobs (uri, evaluation_name, metric) VALUES (?, ?, ?)",
                        (uri, evaluation_name, metric)
                    )
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding jobs for policy: {e}")
            raise

    def get_next_job(self):
        """
        Get the next job from the queue.
        
        Returns:
            A dictionary with job details, or None if queue is empty
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")
            cursor = self.conn.execute("""
                SELECT id, uri, evaluation_name, metric
                FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                job_id, uri, evaluation_name, metric = row
                
                # Mark job as in progress
                self.conn.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,)
                )
                
                # Get policy details
                cursor = self.conn.execute(
                    "SELECT name, version FROM policies WHERE uri = ?",
                    (uri,)
                )
                name, version = cursor.fetchone()
                
                self.conn.commit()
                
                return {
                    'id': job_id,
                    'uri': uri,
                    'name': name,
                    'version': version,
                    'evaluation_name': evaluation_name,
                    'metric': metric
                }
            
            self.conn.commit()
            return None
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error getting next job: {e}")
            raise

    def complete_job(self, job_id: int, success: bool = True):
        """
        Mark a job as complete.
        
        Args:
            job_id: The job ID
            success: Whether the job completed successfully
        """
        try:
            status = 'completed' if success else 'failed'
            self.conn.execute(
                "UPDATE jobs SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, job_id)
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error completing job: {e}")
            raise

    def list_jobs(self, status: str = None):
        """
        List jobs in the queue.
        
        Args:
            status: Optional status filter
            
        Returns:
            DataFrame with job details
        """
        if status:
            sql = """
            SELECT j.id, j.uri, p.name, p.version, j.evaluation_name, j.metric, j.status, 
                   j.created_at, j.started_at, j.completed_at
            FROM jobs j
            JOIN policies p ON j.uri = p.uri
            WHERE j.status = ?
            ORDER BY j.created_at ASC
            """
            return self.query(sql, (status,))
        else:
            sql = """
            SELECT j.id, j.uri, p.name, p.version, j.evaluation_name, j.metric, j.status, 
                   j.created_at, j.started_at, j.completed_at
            FROM jobs j
            JOIN policies p ON j.uri = p.uri
            ORDER BY j.created_at ASC
            """
            return self.query(sql)