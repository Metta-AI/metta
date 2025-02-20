import os
import json
import duckdb
import wandb
import pandas as pd
from typing import Optional, Dict, Any, List
import logging
import shutil
import gzip

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
        # Average over unique policy_name while maintaining eval_name and episode_index
        df_per_episode = long_df.pivot(index='episode_index', columns=['eval_name', 'policy_name'], values='total_metric').fillna(0)
        metric_df = df_per_episode.copy()
        if not group_by_episode:
            stats_df = df_per_episode.agg(['mean', 'std'])
            metric_df = stats_df.T.reset_index()
            metric_df.columns = ['eval_name', 'policy_name', 'mean', 'std']
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

        cache_dir = os.path.join(wandb.env.get_cache_dir(), "artifacts", self.artifact_identifier)
        artifact_versions = self.api.artifacts(type_name=artifact_name,name=self.artifact_identifier)
        artifact_dirs = [os.path.join(cache_dir, f"v{v}") for v in range(len(artifact_versions))]
        for dir, artifact in zip(artifact_dirs, artifact_versions):
            if not os.path.exists(dir):
                artifact.download(root=dir)
                with gzip.open(dir, "rb") as f_in:
                    with open(dir.replace('.gz', ''), "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        logger.info(f"Loaded {len(artifact_dirs)} artifacts")
        all_records = []
        for artifact_dir in artifact_dirs:
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith('.json')]
            if len(json_files) != 1:
                raise FileNotFoundError(f"Expected exactly one JSON file in {artifact_dir}, found {len(json_files)}")
            json_path = os.path.join(artifact_dir, json_files[0])
            with open(json_path, "r") as f:
                data = json.load(f)
            data = self._prepare_data(data)
            all_records.extend(data)

        df = pd.DataFrame(all_records)
        super().__init__(df)
