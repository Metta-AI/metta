import os
import json
import duckdb
import wandb
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

from rl.eval.queries import all_fields, total_metric


logger = logging.getLogger("eval_stats_db.py")

class EvalStatsDB:
    def __init__(self, data: pd.DataFrame):
        self._db = duckdb.connect(database=':memory:')
        self._table_name = "eval_data"
        self._db.register(self._table_name, data)
        self.available_metrics = self._query(all_fields())
        logger.info(f"Loaded {len(self.available_metrics)} metrics from {self._table_name}")

    @staticmethod
    def _prepare_data(data) -> List[dict]:
        """
        Each record is augmented with:
          - 'episode_index': the index of the episode.
          - 'agent_index': the index of the record within the episode.

          data: list (per episode) of lists of dicts (per agent)
          eg [
                [{'action.use ': 1.0, 'r2.gained': 100}, {'action.use ': 3.0, 'r2.gained': 200}]
             ...
                [{'action.use ': 1.0, 'r2.gained': 100}, {'action.use ': 3.0, 'r2.gained': 200}]
             ]

          flattened: list of dicts, where each dict is a record from a single episode.
          eg [
            {'episode_index': 0, 'agent_index': 0, 'action.use ': 1.0, 'r2.gained': 100}
            ...
            {'episode_index': n, 'agent_index': n, 'action.use ': 1.0, 'r2.gained': 100}
          ]
        """
        flattened = []
        for episode_index, episode in enumerate(data):
            for agent_index, record in enumerate(episode):
                record["episode_index"] = episode_index
                record["agent_index"] = agent_index
                flattened.append(record)
        return flattened

    def _query(self, sql_query: str) -> pd.DataFrame:
        try:
            result = self._db.execute(sql_query).fetchdf()
            return result
        except Exception as e:
            raise RuntimeError(f"SQL query failed: {sql_query}\nError: {e}")

    def _metric(self, metric_field: str, filters: Optional[Dict[str, Any]] = None, group_by_episode: bool = False) -> pd.DataFrame:
        long_df = self._query(total_metric(metric_field, filters))
        df_per_episode = long_df.pivot(index='episode_index', columns='policy_name', values='total_metric').fillna(0)
        metric_df = df_per_episode.copy()
        if not group_by_episode:
            mean_series = df_per_episode.mean(axis=0)
            std_series = df_per_episode.std(axis=0)
            metric_df = pd.DataFrame({
                f'{metric_field}_mean': mean_series,
                f'{metric_field}_std': std_series
            })

        return df_per_episode, metric_df

    @staticmethod
    def from_uri(uri: str, wandb_run = None):
        if uri.startswith("wandb://"):
            artifact_name = uri.split("/")[-1]
            return EvalStatsDbWandb(artifact_name, wandb_run)
        else:
            if uri.startswith("file://"):
                json_path = uri.split("file://")[1]
            else:
                json_path = uri
            json_path = json_path if json_path.endswith('.json') else json_path + '.json'
            return EvalStatsDbFile(json_path)


class EvalStatsDbFile(EvalStatsDB):
    """
    Database for loading eval stats from a file.
    """
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        data = self._prepare_data(data)

        super().__init__(pd.DataFrame(data))

class EvalStatsDbWandb(EvalStatsDB):
    """
    Database for loading eval stats from wandb.
    """
    def __init__(self, artifact_name: str, wandb_run):
        self.api = wandb.Api()
        self.artifact_identifier = os.path.join(wandb_run.entity, wandb_run.project, artifact_name)

        artifact_versions = self.api.artifacts(
            type_name=artifact_name,
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
                logger.info(f"Warning: Failed to load version {artifact.id}: {e}")
        df = pd.DataFrame(all_records)
        super().__init__(df)
