from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

T = TypeVar("T", bound=BaseModel)


class NotAuthenticatedError(Exception):
    pass


def get_machine_token(stats_server_uri: str | None = None) -> str | None:
    """Get machine token for the given stats server.

    Args:
        stats_server_uri: The stats server URI to get token for.

    Returns:
        The machine token or None if not found.
    """
    if not stats_server_uri:
        return None

    # Use the same authenticator pattern as the login script
    token = observatory_auth_config.load_token(stats_server_uri)

    if not token or token.lower() == "none":
        return None

    return token


class BaseAppBackendClient:
    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None) -> None:
        self._http_client = httpx.Client(
            base_url=backend_url,
            timeout=30.0,
        )

        self._machine_token = machine_token or get_machine_token(backend_url)

    def close(self):
        self._http_client.close()

    def _make_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    def _validate_authenticated(self) -> str:
        from metta.app_backend.server import WhoAmIResponse

        auth_user = self._make_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise NotAuthenticatedError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email
