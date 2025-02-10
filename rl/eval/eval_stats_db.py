import os
import json
import duckdb
import wandb
from wandb.sdk import wandb_run
import pandas as pd
from typing import Optional, Dict, Any, List
from omegaconf import OmegaConf

class EvalStatsDB:
    def __init__(self, table_name: str = "eval_data"):
        self.table_name = table_name

    def load_df(self):
        pass

    def load_and_register(self):
        df = self.load_df()
        self.con.register(self.table_name, df)

    def restructure_data(self, data) -> List[dict]:
        """
        Each record is augmented with:
          - 'episode_index': the index of the episode.
          - 'agent_index': the index of the record within the episode.
        """
        if isinstance(data, list) and data and isinstance(data[0], list):
            flattened = []
            for episode_index, episode in enumerate(data):
                if isinstance(episode, list):
                    for agent_index, record in enumerate(episode):
                        if isinstance(record, dict):
                            record["episode_index"] = episode_index
                            record["agent_index"] = agent_index
                        flattened.append(record)
                else:
                    flattened.append(episode)
            return flattened
        return data


    def get_metrics_by_pattern(self, pattern: str) -> List[str]:
        """
        Retrieve all metric fields that contain the given pattern.
        """
        schema_query = f"PRAGMA table_info({self.table_name});"
        schema_df = self.query(schema_query)
        all_columns = schema_df['name'].tolist()
        metric_fields = [col for col in all_columns if pattern in col]
        return metric_fields

    def query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a SQL query on the loaded artifact table and return the results as a Pandas DataFrame.
        """
        try:
            result = self.con.execute(sql_query).fetchdf()
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
            FROM {self.table_name}
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

class EvalStatsDbFile(EvalStatsDB):
    def __init__(self, file_path: str, table_name: str = "eval_data"):
        super().__init__(table_name)
        self.file_path = file_path
        self.con = duckdb.connect(database=':memory:')

        self.load_and_register()

    def load_df(self):
        print(f"Loading file: {self.file_path}")
        with open(self.file_path, "r") as f:
            data = json.load(f)
        data = self.restructure_data(data)

        df = pd.DataFrame(data)
        return df

class EvalStatsDbWandb(EvalStatsDB):
    def __init__(self, entity: str, project: str, artifact_name: str, version = None, table_name: str = "eval_data"):
        """
        Initialize the instance by downloading the wandb artifact(s) and loading their JSON data into DuckDB.

        Args:
            entity (str): Your wandb username or team name.
            project (str): The name of your wandb project.
            artifact_name (str): The name of the artifact.
            alias (str, optional): The alias of the artifact (e.g., "latest"). If None, all versions will be loaded.
            table_name (str, optional): The name to assign the DuckDB table. Defaults to "artifact_table".
        """
        super().__init__(table_name)
        self.entity = entity
        self.project = project
        self.artifact_name = artifact_name
        self.version = version
        self.api = wandb.Api()
        self.con = duckdb.connect(database=':memory:')

        self.artifact_identifier = f"{self.entity}/{self.project}/{self.artifact_name}"

        self.load_and_register()

    def get_versions(self):
        if self.version is not None:
            if isinstance(self.version, str):
                return [self.api.artifact(f"{self.artifact_identifier}:{self.version}")]
            elif isinstance(self.version, list):
                return [self.api.artifact(f"{self.artifact_identifier}:{v}") for v in self.version]
            else:
                raise ValueError("Version must be a string or a list of strings")

        return self.api.artifacts(
            type_name=self.artifact_name,
            name=self.artifact_identifier
        )

    def load_df(self):
        versions = self.get_versions()
        all_records = []
        for artifact in versions:
            artifact_dir = artifact.download()
            json_path = os.path.join(artifact_dir, f"policy.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found at expected path: {json_path}")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                data = self.restructure_data(data)
                version_info = artifact.id
                for record in data:
                    record["artifact_version"] = version_info
                all_records.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load version {artifact.id}: {e}")
        df = pd.DataFrame(all_records)
        return df
