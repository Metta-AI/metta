import typing

import httpx
import pydantic

import metta.common.auth.auth_config_reader_writer
import metta.common.util.collections
import metta.common.util.constants

T = typing.TypeVar("T", bound=pydantic.BaseModel)
ClientT = typing.TypeVar("ClientT", bound="BaseAppBackendClient")


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
    token = metta.common.auth.auth_config_reader_writer.observatory_auth_config.load_token(stats_server_uri)

    if not token or token.lower() == "none":
        return None

    return token


class BaseAppBackendClient:
    def __init__(
        self, backend_url: str = metta.common.util.constants.PROD_STATS_SERVER_URI, machine_token: str | None = None
    ) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=backend_url,
            timeout=30.0,
        )

        self._machine_token = machine_token or get_machine_token(backend_url)

    async def __aenter__(self):
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any
    ) -> None:
        await self.close()

    async def close(self):
        await self._http_client.aclose()

    async def _make_request(self, response_type: typing.Type[T], method: str, url: str, **kwargs):
        headers = metta.common.util.collections.remove_none_values({"X-Auth-Token": self._machine_token})
        response = await self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    async def _validate_authenticated(self) -> str:
        import metta.app_backend.server

        auth_user = await self._make_request(metta.app_backend.server.WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise NotAuthenticatedError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email
