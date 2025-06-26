import os
from logging import Logger

from httpx import Client
from omegaconf import DictConfig, ListConfig

from app_backend.stats_client import StatsClient


def get_machine_token() -> str | None:
    env_token = os.getenv("METTA_API_KEY")
    if env_token is not None:
        return env_token
    token_file = os.path.expanduser("~/.metta/observatory_token")
    if os.path.exists(token_file):
        return open(token_file).read().strip()

    return None


def get_stats_client(cfg: DictConfig | ListConfig, logger: Logger) -> StatsClient | None:
    if isinstance(cfg, DictConfig):
        stats_server_uri: str | None = cfg.get("stats_server_uri", None)
        machine_token = get_machine_token()

        if stats_server_uri is not None and machine_token is not None:
            logger.info(f"Using stats client at {stats_server_uri}")
            http_client = Client(base_url=stats_server_uri)
            return StatsClient(http_client=http_client, machine_token=machine_token)
        else:
            if stats_server_uri is None:
                logger.warning("No stats server URI provided, running without stats collection")
            if machine_token is None:
                logger.warning(
                    "No machine token provided, running without stats collection. "
                    + "You can set METTA_API_KEY or ~/.metta/observatory_token to enable stats collection."
                )
    return None
