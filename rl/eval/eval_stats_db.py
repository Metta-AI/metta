import datetime
import os
import json
import duckdb
import wandb
import pandas as pd
from typing import Optional, Dict, Any, List
from omegaconf import OmegaConf
class EvalStatsDB:
    def __init__(self, data: pd.DataFrame):
        self._db = duckdb.connect(database=':memory:')
        self._db.register(self.TABLE_NAME, data)
        self._table_name = "eval_data"

    def _prepare_data(self, data) -> List[dict]:
        """
        Each record is augmented with:
          - 'episode_index': the index of the episode.
          - 'agent_index': the index of the record within the episode.
          # xcxc: document with an example
        """
        flattened = []
        for episode_index, episode in enumerate(data):
            for agent_index, record in enumerate(episode):
                record["episode_index"] = episode_index
                record["agent_index"] = agent_index
                flattened.append(record)
        return flattened

    def get_metrics_by_pattern(self, pattern: str) -> List[str]:
        """
        Retrieve all metric fields that contain the given pattern.
        """
        schema_query = f"PRAGMA table_info({self.TABLE_NAME});"
        schema_df = self.query(schema_query)
        all_columns = schema_df['name'].tolist()
        metric_fields = [col for col in all_columns if pattern in col]
        return metric_fields

    def query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a SQL query on the loaded artifact table and return the results as a Pandas DataFrame.
        """
        try:
            result = self._db.execute(sql_query).fetchdf()
            return result
        except Exception as e:
            raise RuntimeError(f"SQL query failed: {sql_query}\nError: {e}")

    def _build_where_clause(self, filters: Dict[str, Any]) -> str:
        """Build WHERE clause from a filters dictionary."""
        if not filters:
            return ""
        conditions = []

        # Convert OmegaConf objects to plain Python types if necessary.
        if OmegaConf.is_config(filters):
            filters = OmegaConf.to_container(filters, resolve=True)

        for field, value in filters.items():
            # If field names contain dots, wrap them in quotes.
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            if '.' in field and not field.startswith('"'):
                field = f'"{field}"'
            if isinstance(value, (list, tuple)):
                formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                conditions.append(f"{field} IN ({', '.join(formatted_values)})")
            elif isinstance(value, str):
                value = value.strip()
                if value.startswith(('>', '<', '=', '!=', '>=', '<=', 'IN', 'BETWEEN', 'IS')):
                    conditions.append(f"{field} {value}")
                else:
                    conditions.append(f"{field} = '{value}'")
            else:
                conditions.append(f"{field} = {value}")
        return f"WHERE {' AND '.join(conditions)}" if conditions else ""

    def metric_per_episode_per_policy(self, metric_field: str, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculate and pivot a specified metric over episodes and policies with optional filtering.

        Args:
            metric_field (str): The metric field to aggregate.
            filters (dict, optional): Filtering conditions.

        Returns:
            pd.DataFrame: Pivot table with episodes as rows and policies as columns.
        """
        where_clause = self._build_where_clause(filters or {})
        query = f"""
            SELECT
                episode_index,
                policy_name,
                SUM(CAST("{metric_field}" AS DOUBLE)) AS total_metric
            FROM {self.TABLE_NAME}
            {where_clause}
            GROUP BY episode_index, policy_name
            ORDER BY episode_index, policy_name;
        """
        long_df = self.query(query)
        pivot_df = long_df.pivot(index='episode_index', columns='policy_name', values='total_metric').fillna(0)
        return pivot_df

    def average_metrics_by_policy(self, metric_fields: List[str], filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculate average metrics by policy with optional filtering.

        Args:
            metric_fields (list): List of metric field names.
            filters (dict, optional): Filtering conditions.

        Returns:
            pd.DataFrame: Combined DataFrame with mean and standard deviation for each metric.
        """
        result_dfs = []
        for metric in metric_fields:
            pivot_df = self.metric_per_episode_per_policy(metric, filters)
            mean_series = pivot_df.mean(axis=0)
            std_series = pivot_df.std(axis=0)
            metric_df = pd.DataFrame({
                f'{metric}_mean': mean_series,
                f'{metric}_std': std_series
            })
            result_dfs.append(metric_df)
        combined_df = pd.concat(result_dfs, axis=1)
        return combined_df


    @staticmethod
    def from_uri(uri: str, run: wandb.Run):
        # xcxc
        if uri.startswith("wandb://"):
            return EvalStatsDbWandb.from_uri(uri)
        elif uri.startswith("file://"):
            return EvalStatsDbFile.from_uri(uri)
        else:
            raise ValueError(f"Unsupported URI: {uri}")

class EvalStatsDbFile(EvalStatsDB):
    """
    Database for loading eval stats from a file.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"Loading file: {self.file_path}")
        with open(self.file_path, "r") as f:
            data = json.load(f)
        data = self._prepare_data(data)
        super().__init__(pd.DataFrame(data))

class EvalStatsDbWandb(EvalStatsDB):
    """
    Database for loading eval stats from wandb.
    xcxc: (artifact_name: str, run: wandb.Run)
    """
    def __init__(self, artifact_name: str, run: wandb.Run, start_time: Optional[datetime.datetime] = None, end_time: Optional[datetime.datetime] = None):
        self.artifact_name = artifact_name
        self.run = run
        self.api = wandb.Api()
        self.artifact_identifier = os.path.join(self.entity, self.project, self.artifact_name)

        artifact_versions = self.api.artifacts(
            type_name=self.artifact_name,
            name=self.artifact_identifier
        )
        all_records = []
        for artifact in artifact_versions:
            artifact_dir = artifact.download()
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith('.json')]
            if len(json_files) != 1:
                raise FileNotFoundError(f"Expected exactly one JSON file in {artifact_dir}, found {len(json_files)}")
            json_path = os.path.join(artifact_dir, json_files[0])
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                data = self._prepare_data(data)
                version_info = artifact.id
                for record in data:
                    record["artifact_version"] = version_info
                all_records.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load version {artifact.id}: {e}")
        df = pd.DataFrame(all_records)
        super().__init__(df)
