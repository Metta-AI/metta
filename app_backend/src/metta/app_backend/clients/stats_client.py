from __future__ import annotations

import logging
import uuid
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.app_backend.clients.base_client import NotAuthenticatedError, get_machine_token
from metta.app_backend.metta_repo import EvalTaskRow, PolicyVersionRow
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TasksResponse
from metta.app_backend.routes.sql_routes import SQLQueryResponse
from metta.app_backend.routes.stats_routes import (
    BulkEpisodeUploadResponse,
    PolicyCreate,
    PolicyVersionCreate,
    UUIDResponse,
)
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

logger = logging.getLogger("stats_client")

T = TypeVar("T", bound=BaseModel)


class StatsClient:
    """Synchronous wrapper around AsyncStatsClient using httpx sync client."""

    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None):
        self._backend_url = backend_url
        self._http_client = httpx.Client(
            base_url=backend_url,
            timeout=30.0,
        )

        self.machine_token = machine_token or get_machine_token(backend_url)
        self._machine_token = self.machine_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        self._http_client.close()

    def _make_sync_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    def _validate_authenticated(self) -> str:
        from metta.app_backend.server import WhoAmIResponse

        auth_user = self._make_sync_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise NotAuthenticatedError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email

    def create_policy(
        self, name: str, attributes: dict[str, Any] | None = None, is_system_policy: bool = False
    ) -> UUIDResponse:
        data = PolicyCreate(name=name, attributes=attributes or {}, is_system_policy=is_system_policy)
        return self._make_sync_request(UUIDResponse, "POST", "/stats/policies", json=data.model_dump(mode="json"))

    def create_policy_version(
        self,
        policy_id: uuid.UUID,
        policy_spec: dict[str, Any],
        git_hash: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> UUIDResponse:
        data = PolicyVersionCreate(git_hash=git_hash, policy_spec=policy_spec, attributes=attributes or {})
        return self._make_sync_request(
            UUIDResponse, "POST", f"/stats/policies/{policy_id}/versions", json=data.model_dump(mode="json")
        )

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionRow:
        return self._make_sync_request(PolicyVersionRow, "GET", f"/stats/policies/versions/{policy_version_id}")

    def create_eval_task(self, request: TaskCreateRequest) -> EvalTaskRow:
        return self._make_sync_request(EvalTaskRow, "POST", "/tasks", json=request.model_dump(mode="json"))

    def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        params = filters.model_dump(mode="json", exclude_none=True) if filters else {}
        return self._make_sync_request(TasksResponse, "GET", "/tasks/all", params=params)

    def sql_query(self, query: str) -> SQLQueryResponse:
        return self._make_sync_request(SQLQueryResponse, "POST", "/sql/query", json={"query": query})

    def bulk_upload_episodes(
        self, duckdb_path: str, s3_bucket: str = "metta-stats", s3_prefix: str = "episode_stats"
    ) -> BulkEpisodeUploadResponse:
        """Upload a DuckDB file containing episode stats to the backend.

        The backend will store the file in S3 and write aggregated episodes to the database.
        """
        with open(duckdb_path, "rb") as f:
            files = {"file": ("episodes.duckdb", f, "application/octet-stream")}
            params = {"s3_bucket": s3_bucket, "s3_prefix": s3_prefix}
            headers = remove_none_values({"X-Auth-Token": self._machine_token})
            response = self._http_client.post(
                "/stats/episodes/bulk_upload", files=files, params=params, headers=headers
            )
            response.raise_for_status()
            return BulkEpisodeUploadResponse.model_validate(response.json())

    def update_policy_version_tags(self, policy_version_id: uuid.UUID, tags: dict[str, str]) -> UUIDResponse:
        """Update tags for a specific policy version in Observatory."""
        return self._make_sync_request(
            UUIDResponse, "PUT", f"/stats/policies/versions/{policy_version_id}/tags", json=tags
        )

    @staticmethod
    def create(stats_server_uri: str) -> "StatsClient":
        machine_token = get_machine_token(stats_server_uri)
        if machine_token is None:
            raise NotAuthenticatedError(f"No machine token found for {stats_server_uri}")
        stats_client = StatsClient(backend_url=stats_server_uri, machine_token=machine_token)
        stats_client._validate_authenticated()
        return stats_client
