from pathlib import Path
from typing import Any, Type, TypeVar

import httpx
import yaml
from pydantic import BaseModel

from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI
from softmax.orchestrator.server import WhoAmIResponse

T = TypeVar("T", bound=BaseModel)
ClientT = TypeVar("ClientT", bound="BaseAppBackendClient")


class NotAuthenticatedError(Exception):
    pass


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


class BaseAppBackendClient:
    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=backend_url,
            timeout=30.0,
        )

        self._machine_token = machine_token or get_machine_token(backend_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    async def close(self):
        await self._http_client.aclose()

    async def _make_request(self, response_type: Type[T], method: str, url: str, **kwargs):
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = await self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    async def _validate_authenticated(self) -> str:
        auth_user = await self._make_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise NotAuthenticatedError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email
