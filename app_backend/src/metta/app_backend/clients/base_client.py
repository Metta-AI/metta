from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

T = TypeVar("T", bound=BaseModel)


class BaseAppBackendClient:
    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None) -> None:
        self._http_client = httpx.AsyncClient(
            base_url=backend_url,
            timeout=30.0,
        )
        from metta.common.util.stats_client_cfg import get_machine_token

        self._machine_token = machine_token or get_machine_token(backend_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    async def close(self):
        await self._http_client.aclose()

    async def _make_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = await self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())
