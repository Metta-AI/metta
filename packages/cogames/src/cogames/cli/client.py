from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TypeVar

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
        **kwargs: Any,
    ) -> T | dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["X-Auth-Token"] = self._token

        response = self._http_client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()

        if response_type is not None:
            return TypeAdapter(response_type).validate_python(response.json())
        return response.json()

    def _get(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("GET", path, response_type, **kwargs)

    def _post(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("POST", path, response_type, **kwargs)

    def _put(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("PUT", path, response_type, **kwargs)

    def get_seasons(self) -> list[SeasonInfo]:
        return self._get("/tournament/seasons", list[SeasonInfo])

    def get_leaderboard(self, season_name: str) -> list[LeaderboardEntry]:
        return self._get(f"/tournament/seasons/{season_name}/leaderboard", list[LeaderboardEntry])

    def lookup_policy_version(
        self,
        name: str,
        version: int | None = None,
    ) -> PolicyVersionInfo | None:
        params: dict[str, Any] = {"name": name}
        if version is not None:
            params["version"] = version
        try:
            return self._get("/stats/policies/my-versions/lookup", PolicyVersionInfo, params=params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionInfo:
        return self._get(f"/stats/policy-versions/{policy_version_id}", PolicyVersionInfo)

    def submit_to_season(self, season_name: str, policy_version_id: uuid.UUID) -> dict[str, Any]:
        return self._post(
            f"/tournament/seasons/{season_name}/submissions",
            json={"policy_version_id": str(policy_version_id)},
        )

    def get_presigned_upload_url(self, name: str) -> dict[str, Any]:
        return self._post("/stats/policies/submit/presigned-url", json={"name": name})

    def complete_policy_upload(self, upload_id: str, name: str, policy_spec: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            "/stats/policies/submit/complete",
            json={"upload_id": upload_id, "name": name, "policy_spec": policy_spec},
        )

    def update_policy_version_tags(self, policy_version_id: uuid.UUID, tags: dict[str, str]) -> dict[str, Any]:
        return self._put(f"/stats/policies/versions/{policy_version_id}/tags", json=tags)
