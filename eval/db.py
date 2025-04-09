"""
Database interface for storing and querying policy evaluation metrics.
"""

import os
import sqlite3
import logging
import pandas as pd
from omegaconf import DictConfig

from typing import Dict, Tuple, List
import hydra
from rl.wandb.wandb_context import WandbContext
from rl.eval.eval_stats_db import EvalStatsDB

logger = logging.getLogger(__name__)

class PolicyEvalDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS policies (
        uri TEXT PRIMARY KEY,
        training_env TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS evaluations (
        name TEXT NOT NULL,
        metric TEXT NOT NULL,
        PRIMARY KEY (name, metric)
    );

    CREATE TABLE IF NOT EXISTS policy_evaluations (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_uri TEXT REFERENCES policies(uri),
        evaluation_name TEXT NOT NULL,
        metric TEXT NOT NULL,
        mean REAL,
        stdev REAL,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (policy_uri, evaluation_name, metric),
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
                    evaluation_name = row['eval_name']
                    
                    # Create policy if needed
                    if policy_uri not in created_policies:   
                        cursor = self.conn.execute("SELECT 1 FROM policies WHERE uri = ?", (policy_uri,))
                        if not cursor.fetchone():
                            self.conn.execute(
                                "INSERT INTO policies (uri) VALUES (?)",
                                (policy_uri,)
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
                    (policy_uri, evaluation_name, metric, mean, stdev) 
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
            
    def get_matrix_data(self, metric: str) -> pd.DataFrame:
        """Get matrix data for heatmap visualization."""
        sql = """
        SELECT 
            p.uri as policy_uri,
            pe.evaluation_name, 
            pe.mean as value
        FROM policy_evaluations pe
        JOIN policies p ON pe.policy_uri = p.uri
        WHERE pe.metric = ?
        ORDER BY p.uri, pe.evaluation_name
        """
        
        df = self.query(sql, (metric,))
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Calculate overall scores (average across evaluations) 
        overall_scores = df.groupby('policy_uri')['value'].mean().reset_index()
        overall_scores['evaluation_name'] = 'Overall'
        combined_df = pd.concat([
            df, 
            overall_scores[['policy_uri', 'evaluation_name', 'value']]
        ])
        
        # Create a proper matrix (policies as rows, evaluations as columns)
        matrix = combined_df.pivot_table(
            index='policy_uri', 
            columns='evaluation_name', 
            values='value'
        )
        
        # Reorder columns to put 'Overall' first
        if 'Overall' in matrix.columns:
            cols = ['Overall'] + [c for c in matrix.columns if c != 'Overall']
            matrix = matrix[cols]
        
        # Reorder rows by overall score
        policy_order = overall_scores.sort_values('value', ascending=True)['policy_uri'].tolist() 
        matrix = matrix.reindex(policy_order)

        matrix.attrs['eval_names'] = {name: name for name in matrix.columns}
        return matrix