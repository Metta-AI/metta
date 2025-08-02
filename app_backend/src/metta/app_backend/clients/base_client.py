import asyncio
import logging
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.app_backend.server import WhoAmIResponse
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

T = TypeVar("T", bound=BaseModel)
ClientT = TypeVar("ClientT", bound="BaseAppBackendClient")


class BaseAppBackendClient:
    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=backend_url,
            timeout=30.0,
        )

        from metta.common.util.stats_client_cfg import get_machine_token

        self._machine_token = machine_token or get_machine_token(backend_url)
        self._logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    async def close(self):
        await self._http_client.aclose()

    async def _make_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = await self._http_client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                return response_type.model_validate(response.json())
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == max_retries:
                    self._logger.warning(f"Request failed after {max_retries + 1} attempts: {e}")
                    raise

                delay = base_delay * (2**attempt)
                self._logger.debug(
                    f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
            except Exception:
                # Don't retry other exceptions like HTTP errors (4xx, 5xx)
                raise

    async def validate_authenticated(self) -> str:
        auth_user = await self._make_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise ConnectionError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email
