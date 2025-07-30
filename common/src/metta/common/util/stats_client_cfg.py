from logging import Logger
from pathlib import Path

import yaml
from httpx import Client
from omegaconf import DictConfig, ListConfig

from metta.app_backend.stats_client import StatsClient
from metta.common.util.constants import PROD_STATS_SERVER_URI


def get_machine_token(stats_server_uri: str | None = None) -> str | None:
    """Get machine token for the given stats server.

    Args:
        stats_server_uri: The stats server URI to get token for.
                         If None, returns token from env var or legacy location.

    Returns:
        The machine token or None if not found.
    """
    yaml_file = Path.home() / ".metta" / "observatory_tokens.yaml"
    if yaml_file.exists():
        with open(yaml_file) as f:
            tokens = yaml.safe_load(f) or {}
        if isinstance(tokens, dict) and stats_server_uri in tokens:
            token = tokens[stats_server_uri].strip()
        else:
            return None
    elif stats_server_uri is None or stats_server_uri in (
        "https://observatory.softmax-research.net/api",
        PROD_STATS_SERVER_URI,
    ):
        # Fall back to legacy token file, which is assumed to contain production
        # server tokens if it exists
        legacy_file = Path.home() / ".metta" / "observatory_token"
        if legacy_file.exists():
            with open(legacy_file) as f:
                token = f.read().strip()
        else:
            return None
    else:
        return None

    if not token or token.lower() == "none":
        return None

    return token


def get_stats_client_direct(stats_server_uri: str | None, logger: Logger) -> StatsClient | None:
    if stats_server_uri is None:
        logger.warning("No stats server URI provided, running without stats collection")
        return None

    machine_token = get_machine_token(stats_server_uri)
    if machine_token is None:
        logger.warning("No machine token provided, running without stats collection")
        return None

    logger.info(f"Using stats client at {stats_server_uri}")
    http_client = Client(base_url=stats_server_uri)
    return StatsClient(http_client=http_client, machine_token=machine_token)


def get_stats_client(cfg: DictConfig | ListConfig, logger: Logger) -> StatsClient | None:
    if not isinstance(cfg, DictConfig):
        return None
    stats_server_uri = cfg.get("stats_server_uri", None)
    return get_stats_client_direct(stats_server_uri, logger)
