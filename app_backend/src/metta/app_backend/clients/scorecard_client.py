from typing import Generic, Literal, TypeVar

from pydantic import RootModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.scorecard_routes import (
    EvalsRequest,
    MetricsRequest,
    PoliciesResponse,
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
