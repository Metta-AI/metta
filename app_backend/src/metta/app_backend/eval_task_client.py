import asyncio
import logging
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCreateRequest,
    TaskResponse,
    TasksResponse,
    TaskUpdateRequest,
    TaskUpdateResponse,
)
from metta.common.util.collections import remove_none_values
from metta.common.util.stats_client_cfg import get_machine_token

T = TypeVar("T", bound=BaseModel)


class EvalTaskClient:
    def __init__(self, backend_url: str) -> None:
        # Configure httpx client with settings that work better with Docker networking
        self._http_client = httpx.AsyncClient(
            base_url=backend_url,
            timeout=30.0,
            # Disable HTTP/2 to avoid potential compatibility issues
            http2=False,
            # Configure limits to avoid connection pooling issues
            limits=httpx.Limits(
                max_keepalive_connections=0,  # Disable keep-alive
                max_connections=10,
            ),
            # Add headers that might help with proxying
            headers={
                "Connection": "close",  # Force connection close after each request
            },
        )
        self._machine_token = get_machine_token(backend_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    async def close(self):
        await self._http_client.aclose()

    async def _make_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        logger = logging.getLogger(__name__)

        # Retry with exponential backoff for connection errors
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await self._http_client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                return response_type.model_validate(response.json())
            except httpx.ConnectError as e:
                # Log more details about connection errors
                logger.error(f"Connection error to {self._http_client.base_url}{url}: {e}")
                logger.error(f"Backend URL: {self._http_client.base_url}")
                logger.error(f"Attempt {attempt + 1}/{max_retries}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
            except httpx.HTTPError as e:
                logger.error(f"HTTP error for {method} {url}: {e}")
                # Don't retry on HTTP errors (4xx, 5xx)
                raise

    async def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        return await self._make_request(TaskResponse, "POST", "/tasks", json=request.model_dump(mode="json"))

    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        return await self._make_request(TasksResponse, "GET", "/tasks/available", params={"limit": limit})

    async def claim_tasks(self, request: TaskClaimRequest) -> TaskClaimResponse:
        return await self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=request.model_dump(mode="json"))

    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        params = {"assignee": assignee} if assignee is not None else {}
        return await self._make_request(TasksResponse, "GET", "/tasks/claimed", params=params)

    async def get_task_by_id(self, task_id: str) -> TaskResponse:
        return await self._make_request(TaskResponse, "GET", f"/tasks/{task_id}")

    async def update_task_status(self, request: TaskUpdateRequest) -> TaskUpdateResponse:
        return await self._make_request(
            TaskUpdateResponse, "POST", "/tasks/claimed/update", json=request.model_dump(mode="json")
        )

    async def get_latest_assigned_task_for_git_hash(self, git_hash: str) -> TaskResponse:
        return await self._make_request(TaskResponse, "GET", "/tasks/latest", params={"git_hash": git_hash})
