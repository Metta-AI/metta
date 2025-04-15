import gzip
import json
import logging
import os
import shutil
from typing import List

import duckdb
import pandas as pd
import wandb

from metta.rl.eval.queries import all_fields

logger = logging.getLogger("eval_stats_db.py")


class EvalStatsDB:
    def __init__(self, data: pd.DataFrame):
        self._db = duckdb.connect(database=":memory:")
        self._table_name = "eval_data"
        self._db.register(self._table_name, data)
        self.available_metrics = self._query(all_fields())
        logger.info(f"Loaded {len(self.available_metrics)} metrics from {self._table_name}")

    @staticmethod
    def _flatten_data_into_records(data) -> List[dict]:
        return [record for env_data in data.values() for episode in env_data for record in episode]

    def _query(self, sql_query: str) -> pd.DataFrame:
        try:
            result = self._db.execute(sql_query).fetchdf()
            return result
        except Exception as e:
            raise RuntimeError(f"SQL query failed: {sql_query}\nError: {e}") from e

    @staticmethod
    def from_uri(uri: str, run_dir: str, wandb_run=None):
        # We want eval stats to be the same for train, analysis and eval for a particular run
        save_dir = run_dir.replace("analyze", "train").replace("eval", "train")
        uri = uri or os.path.join(save_dir, "eval_stats")
        if uri.startswith("wandb://"):
            artifact_name = uri.split("/")[-1]
            return EvalStatsDbWandb(artifact_name, wandb_run)
        else:
            if uri.startswith("file://"):
                json_path = uri.split("file://")[1]
            else:
                json_path = uri
            if json_path.endswith(".json"):
                json_path = json_path + ".gz"
            elif not json_path.endswith(".json.gz"):
                json_path = json_path + ".json.gz"
            return EvalStatsDbFile(json_path)


class EvalStatsDbFile(EvalStatsDB):
    """
    Database for loading eval stats from a file.
    """

    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with gzip.open(json_path, "rt") as f:
            data = json.load(f)
        logger.info(f"Loading eval stats from {json_path}")

        data = self._flatten_data_into_records(data)

        super().__init__(pd.DataFrame(data))


class EvalStatsDbWandb(EvalStatsDB):
    """
    Database for loading eval stats from wandb.
    """

    def __init__(self, artifact_name: str, wandb_run, from_cache: bool = False):
        self.api = wandb.Api()
        self.artifact_identifier = os.path.join(wandb_run.entity, wandb_run.project, artifact_name)

        cache_dir = os.path.join(wandb.env.get_cache_dir(), "artifacts", self.artifact_identifier)
        artifact_versions = self.api.artifacts(type_name=artifact_name, name=self.artifact_identifier)
        artifact_dirs = [os.path.join(cache_dir, v.name) for v in artifact_versions]
        for dir, artifact in zip(artifact_dirs, artifact_versions, strict=False):
            if os.path.exists(dir) and from_cache:
                logger.info(f"Loading from cache: {dir}")
            else:
                artifact.download(root=dir)
                path = os.path.join(dir, os.listdir(dir)[0])
                if path.endswith(".json.gz"):
                    with gzip.open(path, "rb") as f_in:
                        with open(path.replace(".gz", ""), "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

        logger.info(f"Loaded {len(artifact_dirs)} artifacts")
        all_records = []
        for artifact_dir in artifact_dirs:
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
            if len(json_files) != 1:
                raise FileNotFoundError(f"Expected exactly one JSON file in {artifact_dir}, found {len(json_files)}")
            json_path = os.path.join(artifact_dir, json_files[0])
            with open(json_path, "r") as f:
                data = json.load(f)
            data = self._flatten_data_into_records(data)
            all_records.extend(data)

        df = pd.DataFrame(all_records)
        super().__init__(df)
