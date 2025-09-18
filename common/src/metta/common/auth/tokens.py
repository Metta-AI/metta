from pathlib import Path

import yaml


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
    else:
        return None

    if not token or token.lower() == "none":
        return None

    return token
