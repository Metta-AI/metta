from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TypeVar, overload

import httpx
from pydantic import BaseModel, TypeAdapter

from cogames.cli.base import console
from cogames.cli.login import CoGamesAuthenticator

T = TypeVar("T")


class PolicyVersionInfo(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    created_at: datetime
    policy_created_at: datetime
    user_id: str
    name: str
    version: int


class PoolInfo(BaseModel):
    name: str
    description: str


class SeasonInfo(BaseModel):
    name: str
    summary: str
    pools: list[PoolInfo]


class LeaderboardEntry(BaseModel):
    rank: int
    policy: PolicyVersionSummary
    score: float
    matches: int


class PolicyVersionSummary(BaseModel):
    id: uuid.UUID
    name: str | None
    version: int | None


class SubmitToSeasonResponse(BaseModel):
    pools: list[str]


class PoolMembership(BaseModel):
    pool_name: str
    active: bool
    completed: int
    failed: int
    pending: int


class SeasonPolicyEntry(BaseModel):
    policy: PolicyVersionSummary
    pools: list[PoolMembership]
    entered_at: str


class TournamentServerClient:
    def __init__(
        self,
        server_url: str,
        token: str | None = None,
        login_server: str | None = None,
    ):
        self._server_url = server_url
        self._token = token
        self._login_server = login_server
        self._http_client = httpx.Client(base_url=server_url, timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        self._http_client.close()

    @classmethod
    def from_login(cls, server_url: str, login_server: str) -> TournamentServerClient | None:
        authenticator = CoGamesAuthenticator()
        if not authenticator.has_saved_token(login_server):
            console.print("[red]Error:[/red] Not authenticated.")
            console.print("Please run: [cyan]cogames login[/cyan]")
            return None

        token = authenticator.load_token(login_server)
        if not token:
            console.print(f"[red]Error:[/red] Token not found for {login_server}")
            return None

        return cls(server_url=server_url, token=token, login_server=login_server)

    def _request(
        self,
        method: str,
        path: str,
        response_type: type[T] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> T | dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["X-Auth-Token"] = self._token

        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self._http_client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()

        if response_type is not None:
            return TypeAdapter(response_type).validate_python(response.json())
        return response.json()

    @overload
    def _get(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _get(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _get(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("GET", path, response_type, **kwargs)

    @overload
    def _post(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _post(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _post(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("POST", path, response_type, **kwargs)

    @overload
    def _put(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _put(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _put(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("PUT", path, response_type, **kwargs)

    def get_seasons(self) -> list[SeasonInfo]:
        return self._get("/tournament/seasons", list[SeasonInfo])

    def get_leaderboard(self, season_name: str) -> list[LeaderboardEntry]:
        return self._get(f"/tournament/seasons/{season_name}/leaderboard", list[LeaderboardEntry])

    def get_my_policy_versions(
        self,
        name: str | None = None,
        version: int | None = None,
    ) -> list[PolicyVersionInfo]:
        params: dict[str, Any] = {"mine": "true", "limit": 100}
        if name is not None:
            params["name_exact"] = name
        if version is not None:
            params["version"] = version
        result = self._get("/stats/policy-versions", params=params)
        entries = result.get("entries", [])
        return [PolicyVersionInfo.model_validate(e) for e in entries]

    def lookup_policy_version(
        self,
        name: str,
        version: int | None = None,
    ) -> PolicyVersionInfo | None:
        versions = self.get_my_policy_versions(name=name, version=version)
        return versions[0] if versions else None

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionInfo:
        return self._get(f"/stats/policy-versions/{policy_version_id}", PolicyVersionInfo)

    def submit_to_season(self, season_name: str, policy_version_id: uuid.UUID) -> SubmitToSeasonResponse:
        return self._post(
            f"/tournament/seasons/{season_name}/submissions",
            SubmitToSeasonResponse,
            json={"policy_version_id": str(policy_version_id)},
        )

    def get_season_policies(self, season_name: str, mine: bool = False) -> list[SeasonPolicyEntry]:
        return self._get(
            f"/tournament/seasons/{season_name}/policies",
            list[SeasonPolicyEntry],
            params={"mine": "true"} if mine else None,
        )

    def get_presigned_upload_url(self) -> dict[str, Any]:
        return self._post("/stats/policies/submit/presigned-url", timeout=60.0)

    def complete_policy_upload(self, upload_id: str, name: str) -> dict[str, Any]:
        return self._post(
            "/stats/policies/submit/complete",
            timeout=120.0,
            json={"upload_id": upload_id, "name": name},
        )

    def update_policy_version_tags(self, policy_version_id: uuid.UUID, tags: dict[str, str]) -> dict[str, Any]:
        return self._put(f"/stats/policies/versions/{policy_version_id}/tags", json=tags)

    def get_policy_memberships(self, policy_version_id: uuid.UUID) -> list[dict[str, Any]]:
        return self._get(f"/tournament/policies/{policy_version_id}/memberships", list[dict[str, Any]])

    def get_my_memberships(self) -> dict[str, list[str]]:
        """Get all season memberships for the user's policy versions.

        Returns a mapping of policy_version_id -> list of season names.
        """
        return self._get("/tournament/my-memberships", dict[str, list[str]])
