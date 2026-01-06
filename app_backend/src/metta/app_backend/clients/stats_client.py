from __future__ import annotations

import logging
import uuid
from typing import Any, TypeVar

import httpx
from pydantic import TypeAdapter

from metta.app_backend.clients.base_client import NotAuthenticatedError, get_machine_token
from metta.app_backend.metta_repo import EvalTaskRow, PolicyVersionWithName
from metta.app_backend.models.job_request import JobRequest, JobRequestCreate, JobRequestUpdate, JobStatus, JobType
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TasksResponse
from metta.app_backend.routes.leaderboard_routes import (
    LeaderboardPoliciesResponse,
)
from metta.app_backend.routes.sql_routes import SQLQueryResponse
from metta.app_backend.routes.stats_routes import (
    BulkEpisodeUploadResponse,
    CompleteBulkUploadRequest,
    EpisodeQueryRequest,
    EpisodeQueryResponse,
    MyPolicyVersionsResponse,
    PolicyCreate,
    PolicyVersionCreate,
    PolicyVersionsResponse,
    PresignedUploadUrlResponse,
    UUIDResponse,
)
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

logger = logging.getLogger("stats_client")

T = TypeVar("T")


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

    def _make_sync_request(self, response_type: type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return TypeAdapter(response_type).validate_python(response.json())

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
        s3_path: str | None = None,
    ) -> UUIDResponse:
        data = PolicyVersionCreate(
            git_hash=git_hash, policy_spec=policy_spec, attributes=attributes or {}, s3_path=s3_path
        )
        return self._make_sync_request(
            UUIDResponse, "POST", f"/stats/policies/{policy_id}/versions", json=data.model_dump(mode="json")
        )

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionWithName:
        return self._make_sync_request(PolicyVersionWithName, "GET", f"/stats/policies/versions/{policy_version_id}")

    def create_eval_task(self, request: TaskCreateRequest) -> EvalTaskRow:
        return self._make_sync_request(EvalTaskRow, "POST", "/tasks", json=request.model_dump(mode="json"))

    def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        params = filters.model_dump(mode="json", exclude_none=True) if filters else {}
        return self._make_sync_request(TasksResponse, "GET", "/tasks/all", params=params)

    def sql_query(self, query: str) -> SQLQueryResponse:
        return self._make_sync_request(SQLQueryResponse, "POST", "/sql/query", json={"query": query})

    def bulk_upload_episodes(self, duckdb_path: str) -> BulkEpisodeUploadResponse:
        """Upload a DuckDB file containing episode stats using presigned URL approach.

        This method:
        1. Requests a presigned URL from the backend
        2. Uploads the DuckDB file directly to S3 using the presigned URL
        3. Notifies the backend to process the uploaded file

        The backend will then process the file from S3 and write aggregated episodes to the database.
        """
        # Step 1: Get presigned URL
        presigned_response = self._make_sync_request(
            PresignedUploadUrlResponse, "POST", "/stats/episodes/bulk_upload/presigned-url"
        )

        # Step 2: Upload file directly to S3 using presigned URL
        with open(duckdb_path, "rb") as f:
            # Use a plain HTTP client for S3 upload (no auth headers needed)
            s3_response = httpx.put(
                presigned_response.upload_url,
                content=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300.0,  # 5 minute timeout for large files
            )
            s3_response.raise_for_status()

        # Step 3: Notify backend to process the uploaded file

        completion_request = CompleteBulkUploadRequest(upload_id=presigned_response.upload_id)
        completion_response = self._make_sync_request(
            BulkEpisodeUploadResponse,
            "POST",
            "/stats/episodes/bulk_upload/complete",
            json=completion_request.model_dump(mode="json"),
        )

        return completion_response

    def update_policy_version_tags(self, policy_version_id: uuid.UUID, tags: dict[str, str]) -> UUIDResponse:
        """Update tags for a specific policy version in Observatory."""
        return self._make_sync_request(
            UUIDResponse, "PUT", f"/stats/policies/versions/{policy_version_id}/tags", json=tags
        )

    def get_leaderboard_policies_v2(self) -> LeaderboardPoliciesResponse:
        return self._make_sync_request(LeaderboardPoliciesResponse, "GET", "/leaderboard/v2")

    def get_leaderboard_policies_with_vor(self) -> LeaderboardPoliciesResponse:
        return self._make_sync_request(LeaderboardPoliciesResponse, "GET", "/leaderboard/v2/vor")

    def get_my_policy_versions(self) -> MyPolicyVersionsResponse:
        return self._make_sync_request(
            MyPolicyVersionsResponse,
            "GET",
            "/stats/policies/my-versions",
        )

    def get_policy_versions(
        self,
        name_exact: str | None = None,
        name_fuzzy: str | None = None,
        version: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PolicyVersionsResponse:
        params = remove_none_values(
            {
                "name_exact": name_exact,
                "name_fuzzy": name_fuzzy,
                "version": version,
                "limit": limit,
                "offset": offset,
            }
        )
        return self._make_sync_request(PolicyVersionsResponse, "GET", "/stats/policy-versions", params=params)

    def get_versions_for_policy(
        self,
        policy_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> PolicyVersionsResponse:
        params = remove_none_values({"limit": limit, "offset": offset})
        return self._make_sync_request(
            PolicyVersionsResponse, "GET", f"/stats/policies/{policy_id}/versions", params=params
        )

    def get_leaderboard_policies_v2_users_me(self) -> LeaderboardPoliciesResponse:
        return self._make_sync_request(
            LeaderboardPoliciesResponse,
            "GET",
            "/leaderboard/v2/users/me",
        )

    def get_leaderboard_policies_v2_for_policy(self, policy_version_id: uuid.UUID) -> LeaderboardPoliciesResponse:
        return self._make_sync_request(
            LeaderboardPoliciesResponse,
            "GET",
            f"/leaderboard/v2/policy/{policy_version_id}",
        )

    def query_episodes(self, request: EpisodeQueryRequest) -> EpisodeQueryResponse:
        return self._make_sync_request(
            EpisodeQueryResponse, "POST", "/stats/episodes/query", json=request.model_dump(mode="json")
        )

    def create_jobs(self, jobs: list[JobRequestCreate]) -> list[uuid.UUID]:
        return self._make_sync_request(
            list[uuid.UUID], "POST", "/jobs/batch", json=[j.model_dump(mode="json") for j in jobs]
        )

    def list_jobs(
        self,
        statuses: list[JobStatus] | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRequest]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if statuses is not None:
            params["statuses"] = [s.value for s in statuses]
        if job_type is not None:
            params["job_type"] = job_type.value
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.get("/jobs", headers=headers, params=params)
        response.raise_for_status()
        return [JobRequest.model_validate(item) for item in response.json()]

    def get_job(self, job_id: uuid.UUID) -> JobRequest:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.get(f"/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        return JobRequest.model_validate(response.json())

    def update_job(
        self,
        job_id: uuid.UUID,
        update: JobRequestUpdate,
    ) -> JobRequest:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.post(f"/jobs/{job_id}", headers=headers, json=update.model_dump(mode="json"))
        response.raise_for_status()
        return JobRequest.model_validate(response.json())

    @staticmethod
    def create(stats_server_uri: str) -> "StatsClient":
        machine_token = get_machine_token(stats_server_uri)
        if machine_token is None:
            raise NotAuthenticatedError(f"No machine token found for {stats_server_uri}")
        stats_client = StatsClient(backend_url=stats_server_uri, machine_token=machine_token)
        stats_client._validate_authenticated()
        return stats_client
