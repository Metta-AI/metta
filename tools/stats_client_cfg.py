from logging import Logger

from omegaconf import DictConfig, ListConfig

from metta.app.http_client import HttpxClient
from metta.app.stats_client import StatsClient


def get_stats_client(cfg: DictConfig | ListConfig, logger: Logger) -> StatsClient | None:
    if isinstance(cfg, DictConfig):
        stats_server_uri: str | None = cfg.get("stats_server_uri", None)
        if stats_server_uri is not None:
            logger.info(f"Using stats client at {stats_server_uri}")
            user = cfg.get("stats_user", "unknown")
            return StatsClient(http_client=HttpxClient(stats_server_uri), user=user)
    return None
