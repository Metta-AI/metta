from typing import Literal

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


class ListModel(RootModel[list]):
    pass


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

    async def get_eval_names(self, training_run_ids: list[str], run_free_policy_ids: list[str]) -> list:
        payload = EvalsRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
        )
        return list(
            await self._make_request(ListModel, "POST", "/scorecard/evals", json=payload.model_dump(mode="json"))
        )

    async def get_available_metrics(
        self, training_run_ids: list[str], run_free_policy_ids: list[str], eval_names: list[str]
    ) -> list:
        payload = MetricsRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
        )
        metrics_tuples = await self._make_request(
            ListModel, "POST", "/scorecard/metrics", json=payload.model_dump(mode="json")
        )
        metric_names = []
        for metric_tuple in metrics_tuples:
            category, metrics = metric_tuple
            metric_names.extend(metrics)
        return metric_names

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
