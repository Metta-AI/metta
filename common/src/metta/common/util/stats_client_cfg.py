import os
from logging import Logger
from pathlib import Path
from urllib.parse import urlparse

import yaml
from httpx import Client
from omegaconf import DictConfig, ListConfig

from metta.app_backend.stats_client import StatsClient

_URI_FALLBACKS = {
    "https://api.observatory.softmax-research.net": "https://observatory.softmax-research.net/api",
}


def get_machine_token(stats_server_uri: str | None = None) -> str | None:
    """Get machine token for the given stats server.

    Args:
        stats_server_uri: The stats server URI to get token for.
                         If None, returns token from env var or legacy location.

    Returns:
        The machine token or None if not found.
    """
    # First check environment variable (takes precedence)
    env_token = os.getenv("METTA_API_KEY")
    if env_token is not None:
        token = env_token
    else:
        # Try YAML file first
        yaml_file = Path.home() / ".metta" / "observatory_tokens.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file) as f:
                    tokens = yaml.safe_load(f) or {}

                if stats_server_uri and isinstance(tokens, dict):
                    # Try exact match first
                    if stats_server_uri in tokens:
                        token = tokens[stats_server_uri]
                    elif (fallback_uri := _URI_FALLBACKS.get(stats_server_uri)) and fallback_uri in tokens:
                        token = tokens[fallback_uri]
                    else:
                        # Try hostname match
                        hostname = urlparse(stats_server_uri).hostname
                        if hostname:
                            for server_uri, token_value in tokens.items():
                                if urlparse(server_uri).hostname == hostname:
                                    token = token_value
                                    break
                            else:
                                # No match found
                                return None
                        else:
                            return None
                else:
                    # No specific server requested, can't determine which token to use
                    return None
            except Exception:
                # Fall back to legacy file if YAML parsing fails
                pass

        # Fall back to legacy token file
        legacy_file = Path.home() / ".metta" / "observatory_token"
        if legacy_file.exists():
            with open(legacy_file) as f:
                token = f.read().strip()
        else:
            return None

    if not token or token.lower() == "none" or len(token.strip()) == 0:
        return None

    return token


def get_stats_client(cfg: DictConfig | ListConfig, logger: Logger) -> StatsClient | None:
    if isinstance(cfg, DictConfig):
        stats_server_uri: str | None = cfg.get("stats_server_uri", None)
        machine_token = get_machine_token(stats_server_uri)

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
                    + f"You can set METTA_API_KEY or save a token for {stats_server_uri} to enable stats collection."
                )
    return None
