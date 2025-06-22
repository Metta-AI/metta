import os
from logging import Logger

from httpx import Client
from omegaconf import DictConfig, ListConfig

from app_backend.stats_client import StatsClient


def get_stats_client(cfg: DictConfig | ListConfig, logger: Logger) -> StatsClient | None:
    if isinstance(cfg, DictConfig):
        stats_server_uri: str | None = cfg.get("stats_server_uri", None)
        machine_token = os.getenv("METTA_API_KEY")

        if stats_server_uri is not None and machine_token is not None:
            logger.info(f"Using stats client at {stats_server_uri}")
            http_client = Client(base_url=stats_server_uri)
            return StatsClient(http_client=http_client, machine_token=machine_token)
    return None
