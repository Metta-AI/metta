from logging import warning
from typing import Any, Dict, Generic, Literal, TypeVar

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
        import requests

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
    ) -> list[str]:
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
        result = await self._make_request(
            ScorecardData, "POST", "/scorecard/scorecard", json=payload.model_dump(mode="json")
        )
        print(result)
        return result

    async def get_scorecard_data(
        self,
        search_term: str | None = None,
        restrict_to_policy_ids: list[str] | None = None,
        restrict_to_metrics: list[str] | None = None,
        restrict_to_policy_names: list[str] | None = None,
        restrict_to_eval_names: list[str] | None = None,
        policy_selector: Literal["best", "latest"] = "best",
        max_policies: int = 30,
        primary_metric: str | None = None,
        include_run_free_policies: bool = False,
        ignore_missing_policies: bool = False,
    ):
        """
        Fetch real evaluation data using the metta HTTP API (same as repo.ts).

        Args:
            client: ScorecardClient instance
            search_term: Search term to filter policies by name
            restrict_to_policy_ids: List of policy IDs to include (e.g., ["123", "456"])
            restrict_to_metrics: List of metrics to include (e.g., ["reward", "heart.get"])
            restrict_to_policy_names: List of policy name filters (e.g., ["relh.skypilot", "daveey.arena.rnd"])
            restrict_to_eval_names: List of specific evaluation names to include (e.g., ["memory/easy"])
            policy_selector: "best" or "latest" policy selection strategy
            max_policies: Maximum number of policies to display
            include_run_free_policies: Whether to include standalone policies

        Returns:
            ScorecardWidget with real data
        """
        if (
            restrict_to_policy_ids == []
            and restrict_to_metrics == []
            and restrict_to_policy_names == []
            and restrict_to_eval_names == []
        ):
            return None

        if not primary_metric:
            if restrict_to_metrics:
                primary_metric = restrict_to_metrics[0]
            else:
                primary_metric = "reward"
        if restrict_to_metrics and primary_metric not in restrict_to_metrics:
            raise ValueError(f"Primary metric {primary_metric} not found in restrict_to_metrics {restrict_to_metrics}")

        if not self._http_client:
            raise ValueError("client is required to fetch scorecard data")

        if search_term:
            policies_data = await self.search_policies(search=search_term)
        else:
            policies_data = await self.get_policies()

        # Find training run IDs that match our training run names
        training_run_ids = []
        run_free_policy_ids = []
        for policy in policies_data.policies:
            if policy.type == "training_run" and (
                (
                    restrict_to_policy_names
                    and any(filter_policy_name == policy.name for filter_policy_name in restrict_to_policy_names)
                )
                or (
                    restrict_to_policy_ids
                    and any(filter_policy_id == policy.id for filter_policy_id in restrict_to_policy_ids)
                )
            ):
                print(policy.type, policy.name, policy.id)
                training_run_ids.append(policy.id)
            elif (
                policy.type == "policy"
                # and include_run_free_policies # FIXME: i'm not sure when to do this exactly
                and (
                    restrict_to_policy_ids
                    and any(filter_policy_id == policy.id for filter_policy_id in restrict_to_policy_ids)
                )
                or (
                    restrict_to_policy_names
                    and any(filter_policy_name == policy.name for filter_policy_name in restrict_to_policy_names)
                )
            ):
                print(policy.type, policy.name, policy.id)
                run_free_policy_ids.append(policy.id)

        if not training_run_ids:
            raise Exception("No training runs found")

        if restrict_to_eval_names:
            # Use the specific eval names provided
            eval_names = restrict_to_eval_names
        else:
            # Get all available evaluations for these policies
            eval_names = await self.get_eval_names(training_run_ids, run_free_policy_ids)
            if not eval_names:
                raise Exception("No evaluations found for selected training runs")
            print(f"Found {len(eval_names)} available evaluations")

        available_metrics = await self.get_available_metrics(training_run_ids, run_free_policy_ids, eval_names)
        if not available_metrics:
            raise Exception("No metrics found for selected policies and evaluations")

        # Filter to requested metrics that actually exist
        valid_metrics = list(
            filter(
                lambda m: (not restrict_to_metrics or m in restrict_to_metrics),
                available_metrics,
            )
        )
        if not valid_metrics:
            print(f"Available metrics: {sorted(available_metrics)}")
            if restrict_to_metrics:
                warning(f"None of the requested metrics {restrict_to_metrics} are available")
            warning(f"Available metrics are: {sorted(available_metrics)}")
            raise Exception("No valid metrics found")

        scorecard_data: ScorecardData = await self.generate_scorecard(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
            metric=primary_metric,
            policy_selector=policy_selector,
        )

        all_policies = training_run_ids + run_free_policy_ids
        if len(all_policies) != len(scorecard_data.policyNames):
            warning(
                f"Number of policies in scorecard data ({len(scorecard_data.policyNames)}) does not match number of"
            )
            all_len = len(training_run_ids) + len(run_free_policy_ids)
            warning(f"policies in your query ({all_len})")
            if not ignore_missing_policies:
                raise Exception("Number of policies in scorecard data does not match number of policies in your query")

        if not scorecard_data.policyNames:
            warning("No scorecard data found in the database for your query:")
            warning(f"  training_run_ids={training_run_ids}")
            warning(f"  run_free_policy_ids={run_free_policy_ids}")
            warning(f"  eval_names={eval_names}")
            warning(f"  primary_metric={primary_metric}")
            raise Exception("No scorecard data found in database for your query")

        cells = self._make_cells_from_scorecard_data(
            scorecard_data=scorecard_data,
            max_policies=max_policies,
            primary_metric=primary_metric,
        )

        return (cells, scorecard_data, valid_metrics, primary_metric)

    def _make_cells_from_scorecard_data(
        self,
        scorecard_data: ScorecardData,
        max_policies: int,
        primary_metric: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        policy_names = list(scorecard_data.policyNames)
        if len(policy_names) > max_policies:
            # Sort by average score and take top N
            avg_scores = scorecard_data.policyAverageScores
            top_policies = sorted(avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True)[:max_policies]

            filtered_cells = {p: scorecard_data.cells[p] for p in top_policies if p in scorecard_data.cells}
            scorecard_data.policyNames = top_policies
            scorecard_data.cells = filtered_cells
            scorecard_data.policyAverageScores = {p: avg_scores[p] for p in top_policies if p in avg_scores}

        cells = {}
        for policy_name in scorecard_data.policyNames:
            cells[policy_name] = {}
            for eval_name in scorecard_data.evalNames:
                cell = scorecard_data.cells.get(policy_name, {}).get(eval_name)
                if cell:
                    cells[policy_name][eval_name] = {
                        "metrics": {primary_metric: cell.value},
                        "replayUrl": cell.replayUrl,
                        "evalName": eval_name,
                    }
                else:
                    cells[policy_name][eval_name] = {
                        "metrics": {primary_metric: 0.0},
                        "replayUrl": None,
                        "evalName": eval_name,
                    }

        return cells
