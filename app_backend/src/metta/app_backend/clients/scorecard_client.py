from typing import Generic, Literal, TypeVar

import requests
from pydantic import RootModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.scorecard_routes import (
    EvalsRequest,
    MetricsRequest,
    PoliciesResponse,
    PoliciesSearchRequest,
    ScorecardData,
    ScorecardRequest,
)
from metta.app_backend.routes.sql_routes import AIQueryRequest, AIQueryResponse, SQLQueryRequest, SQLQueryResponse

T = TypeVar("T")


class ListModel(RootModel[list[T]], Generic[T]):
    @classmethod
    def model_validate(cls, obj) -> list[T]:
        # Use RootModel's validation to ensure we get a proper list
        instance = super().model_validate(obj)
        return instance.root

    def to_list(self) -> list[T]:
        return self.root


class ScorecardClient(BaseAppBackendClient):
    async def get_policies(self):
        return await self._make_request(PoliciesResponse, "GET", "/scorecard/policies")

    async def search_policies(
        self,
        search: str | None = None,
        policy_type: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PoliciesResponse:
        """Search policies with filtering and pagination.

        Args:
            search: Search term for policy names (case-insensitive partial match)
            policy_type: Filter by policy type ('training_run' or 'policy')
            tags: Filter by tags (policies must have at least one matching tag)
            user_id: Filter by user ID
            limit: Maximum number of results (1-1000)
            offset: Number of results to skip

        Returns:
            PoliciesResponse containing matching policies
        """
        payload = PoliciesSearchRequest(
            search=search,
            policy_type=policy_type,
            tags=tags,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        return await self._make_request(
            PoliciesResponse, "POST", "/scorecard/policies/search", json=payload.model_dump(mode="json")
        )

    def search_policies_sync(
        self,
        search: str | None = None,
        policy_type: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PoliciesResponse:
        """Synchronous version of search_policies for use in widgets and Jupyter notebooks.

        Args:
            search: Search term for policy names (case-insensitive partial match)
            policy_type: Filter by policy type ('training_run' or 'policy')
            tags: Filter by tags (policies must have at least one matching tag)
            user_id: Filter by user ID
            limit: Maximum number of results (1-1000)
            offset: Number of results to skip

        Returns:
            PoliciesResponse containing matching policies
        """
        # Create the request payload
        payload = PoliciesSearchRequest(
            search=search,
            policy_type=policy_type,
            tags=tags,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        # Get the base URL from the async client
        base_url = str(self._http_client.base_url)
        url = f"{base_url}/scorecard/policies/search"

        # Build headers
        from metta.common.util.collections import remove_none_values

        headers = remove_none_values({"X-Auth-Token": self._machine_token, "Content-Type": "application/json"})

        # Make synchronous request using requests library
        response = requests.post(url, json=payload.model_dump(mode="json"), headers=headers, timeout=10)

        # Check for errors
        response.raise_for_status()

        # Parse response
        return PoliciesResponse.model_validate(response.json())

    async def sql_query(self, sql: str):
        payload = SQLQueryRequest(
            query=sql,
        )
        return await self._make_request(SQLQueryResponse, "POST", "/sql/query", json=payload.model_dump(mode="json"))

    async def generate_ai_query(self, description: str):
        payload = AIQueryRequest(
            description=description,
        )
        return await self._make_request(
            AIQueryResponse, "POST", "/sql/generate-query", json=payload.model_dump(mode="json")
        )

    async def get_eval_names(self, training_run_ids: list[str], run_free_policy_ids: list[str]) -> list[str]:
        payload = EvalsRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
        )
        return await self._make_request(
            ListModel[str], "POST", "/scorecard/evals", json=payload.model_dump(mode="json")
        )  # type: ignore

    async def get_available_metrics(
        self, training_run_ids: list[str], run_free_policy_ids: list[str], eval_names: list[str]
    ) -> list:
        payload = MetricsRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
        )
        return await self._make_request(
            ListModel[str], "POST", "/scorecard/metrics", json=payload.model_dump(mode="json")
        )  # type: ignore

    async def generate_scorecard(
        self,
        training_run_ids: list[str],
        run_free_policy_ids: list[str],
        eval_names: list[str],
        metric: str,
        policy_selector: Literal["best", "latest"] = "best",
    ) -> ScorecardData:
        payload = ScorecardRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
            metric=metric,
            training_run_policy_selector=policy_selector,
        )
        return await self._make_request(
            ScorecardData, "POST", "/scorecard/scorecard", json=payload.model_dump(mode="json")
        )
