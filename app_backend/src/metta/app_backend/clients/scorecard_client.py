import uuid
from logging import warning
from typing import Any, Dict, Generic, Literal, TypeVar

import requests
from pydantic import RootModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.dashboard_routes import (
    SavedDashboardCreate,
    SavedDashboardDeleteResponse,
    SavedDashboardListResponse,
    SavedDashboardResponse,
)
from metta.app_backend.routes.entity_routes import (
    TrainingRunDescriptionUpdate,
    TrainingRunListResponse,
    TrainingRunPolicy,
    TrainingRunPolicyListResponse,
    TrainingRunResponse,
    TrainingRunTagsUpdate,
)
from metta.app_backend.routes.eval_task_routes import (
    GitHashesRequest,
    GitHashesResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCreateRequest,
    TaskResponse,
    TasksResponse,
    TaskUpdateRequest,
    TaskUpdateResponse,
)
from metta.app_backend.routes.leaderboard_routes import (
    LeaderboardCreateOrUpdate,
    LeaderboardDeleteResponse,
    LeaderboardListResponse,
    LeaderboardResponse,
)
from metta.app_backend.routes.score_routes import PolicyScoresData, PolicyScoresRequest
from metta.app_backend.routes.scorecard_routes import (
    EvalsRequest,
    LeaderboardScorecardRequest,
    MetricsRequest,
    PoliciesResponse,
    PoliciesSearchRequest,
    ScorecardData,
    ScorecardRequest,
    TrainingRunScorecardRequest,
)
from metta.app_backend.routes.sql_routes import (
    AIQueryRequest,
    AIQueryResponse,
    SQLQueryRequest,
    SQLQueryResponse,
    TableInfo,
    TableSchema,
)
from metta.app_backend.routes.stats_routes import (
    EpisodeCreate,
    EpisodeResponse,
    EpochCreate,
    EpochResponse,
    PolicyCreate,
    PolicyIdResponse,
    PolicyResponse,
    TrainingRunCreate,
)
from metta.app_backend.routes.stats_routes import (
    TrainingRunResponse as StatsTrainingRunResponse,
)
from metta.app_backend.routes.sweep_routes import (
    RunIdResponse,
    SweepCreateRequest,
    SweepCreateResponse,
    SweepInfo,
)
from metta.app_backend.routes.token_routes import (
    TokenCreate,
    TokenListResponse,
    TokenResponse,
)

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

    async def list_tables(self) -> list[TableInfo]:
        """List all available tables in the database (excluding migrations)."""
        return await self._make_request(ListModel[TableInfo], "GET", "/sql/tables")  # type: ignore

    async def get_table_schema(self, table_name: str) -> TableSchema:
        """Get the schema for a specific table."""
        return await self._make_request(TableSchema, "GET", f"/sql/tables/{table_name}/schema")

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

    # Dashboard Routes
    async def list_saved_dashboards(self) -> SavedDashboardListResponse:
        """List all saved dashboards."""
        return await self._make_request(SavedDashboardListResponse, "GET", "/dashboard/saved")

    async def get_saved_dashboard(self, dashboard_id: str) -> SavedDashboardResponse:
        """Get a specific saved dashboard by ID."""
        return await self._make_request(SavedDashboardResponse, "GET", f"/dashboard/saved/{dashboard_id}")

    async def create_saved_dashboard(self, dashboard_data: SavedDashboardCreate) -> SavedDashboardResponse:
        """Create a new saved dashboard."""
        return await self._make_request(
            SavedDashboardResponse, "POST", "/dashboard/saved", json=dashboard_data.model_dump(mode="json")
        )

    async def update_saved_dashboard(
        self, dashboard_id: str, dashboard_state: Dict[str, Any]
    ) -> SavedDashboardResponse:
        """Update an existing saved dashboard."""
        return await self._make_request(
            SavedDashboardResponse, "PUT", f"/dashboard/saved/{dashboard_id}", json=dashboard_state
        )

    async def delete_saved_dashboard(self, dashboard_id: str) -> SavedDashboardDeleteResponse:
        """Delete a saved dashboard."""
        return await self._make_request(SavedDashboardDeleteResponse, "DELETE", f"/dashboard/saved/{dashboard_id}")

    # Entity Routes (Training Runs)
    async def get_training_runs(self) -> TrainingRunListResponse:
        """Get all training runs."""
        return await self._make_request(TrainingRunListResponse, "GET", "/training-runs")

    async def get_training_run(self, run_id: str) -> TrainingRunResponse:
        """Get a specific training run by ID."""
        return await self._make_request(TrainingRunResponse, "GET", f"/training-runs/{run_id}")

    async def update_training_run_description(self, run_id: str, description: str) -> TrainingRunResponse:
        """Update the description of a training run."""
        payload = TrainingRunDescriptionUpdate(description=description)
        return await self._make_request(
            TrainingRunResponse, "PUT", f"/training-runs/{run_id}/description", json=payload.model_dump(mode="json")
        )

    async def update_training_run_tags(self, run_id: str, tags: list[str]) -> TrainingRunResponse:
        """Update the tags of a training run."""
        payload = TrainingRunTagsUpdate(tags=tags)
        return await self._make_request(
            TrainingRunResponse, "PUT", f"/training-runs/{run_id}/tags", json=payload.model_dump(mode="json")
        )

    async def get_training_run_policies(self, run_id: str) -> TrainingRunPolicyListResponse:
        """Get policies for a training run with epoch information."""
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.get(f"/training-runs/{run_id}/policies", headers=headers)
        response.raise_for_status()
        data = response.json()

        # Handle both response formats: raw list or wrapped object
        if isinstance(data, list):
            # Raw list format - convert to our expected format
            policies = [TrainingRunPolicy.model_validate(policy) for policy in data]
            return TrainingRunPolicyListResponse(policies=policies)
        else:
            # Wrapped object format - use model validation
            return TrainingRunPolicyListResponse.model_validate(data)

    # Evaluation Task Routes
    async def create_task(self, task_data: TaskCreateRequest) -> TaskResponse:
        """Create a new evaluation task."""
        return await self._make_request(TaskResponse, "POST", "/tasks", json=task_data.model_dump(mode="json"))

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> TaskResponse | None:
        """Get the latest assigned task for a worker."""
        return await self._make_request(TaskResponse, "GET", f"/tasks/latest?assignee={assignee}")

    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        """Get available tasks."""
        return await self._make_request(TasksResponse, "GET", f"/tasks/available?limit={limit}")

    async def claim_tasks(self, tasks: list[uuid.UUID], assignee: str) -> TaskClaimResponse:
        """Claim tasks."""
        payload = TaskClaimRequest(tasks=tasks, assignee=assignee)
        return await self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=payload.model_dump(mode="json"))

    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        """Get claimed tasks."""
        url = "/tasks/claimed"
        if assignee:
            url += f"?assignee={assignee}"
        return await self._make_request(TasksResponse, "GET", url)

    async def get_git_hashes_for_workers(self, assignees: list[str]) -> GitHashesResponse:
        """Get git hashes for workers."""
        payload = GitHashesRequest(assignees=assignees)
        return await self._make_request(
            GitHashesResponse, "POST", "/tasks/git-hashes", json=payload.model_dump(mode="json")
        )

    async def get_all_tasks(
        self,
        limit: int = 500,
        statuses: list[str] | None = None,
        git_hash: str | None = None,
        policy_ids: list[uuid.UUID] | None = None,
        sim_suites: list[str] | None = None,
    ) -> TasksResponse:
        """Get all tasks with optional filtering."""
        params: dict[str, Any] = {"limit": limit}
        if statuses:
            params["statuses"] = statuses
        if git_hash:
            params["git_hash"] = git_hash
        if policy_ids:
            params["policy_ids"] = [str(pid) for pid in policy_ids]
        if sim_suites:
            params["sim_suites"] = sim_suites

        query_string = "&".join(
            [
                f"{k}={v}" if not isinstance(v, list) else "&".join([f"{k}={item}" for item in v])
                for k, v in params.items()
            ]
        )
        return await self._make_request(TasksResponse, "GET", f"/tasks/all?{query_string}")

    async def update_task_statuses(
        self, updates: dict[uuid.UUID, Any], require_assignee: str | None = None
    ) -> TaskUpdateResponse:
        """Update task statuses."""
        payload = TaskUpdateRequest(updates=updates, require_assignee=require_assignee)
        return await self._make_request(
            TaskUpdateResponse, "POST", "/tasks/claimed/update", json=payload.model_dump(mode="json")
        )

    # Leaderboard Routes
    async def list_leaderboards(self) -> LeaderboardListResponse:
        """List all leaderboards."""
        return await self._make_request(LeaderboardListResponse, "GET", "/leaderboards")

    async def get_leaderboard(self, leaderboard_id: str) -> LeaderboardResponse:
        """Get a specific leaderboard by ID."""
        return await self._make_request(LeaderboardResponse, "GET", f"/leaderboards/{leaderboard_id}")

    async def create_leaderboard(self, leaderboard_data: LeaderboardCreateOrUpdate) -> LeaderboardResponse:
        """Create a new leaderboard."""
        return await self._make_request(
            LeaderboardResponse, "POST", "/leaderboards", json=leaderboard_data.model_dump(mode="json")
        )

    async def update_leaderboard(
        self, leaderboard_id: str, leaderboard_data: LeaderboardCreateOrUpdate
    ) -> LeaderboardResponse:
        """Update a leaderboard."""
        return await self._make_request(
            LeaderboardResponse, "PUT", f"/leaderboards/{leaderboard_id}", json=leaderboard_data.model_dump(mode="json")
        )

    async def delete_leaderboard(self, leaderboard_id: str) -> LeaderboardDeleteResponse:
        """Delete a leaderboard."""
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.delete(f"/leaderboards/{leaderboard_id}", headers=headers)
        response.raise_for_status()
        data = response.json()
        return LeaderboardDeleteResponse.model_validate(data)

    # Score Routes
    async def get_policy_scores(
        self, policy_ids: list[uuid.UUID], eval_names: list[str], metrics: list[str]
    ) -> PolicyScoresData:
        """Get policy scores for given policies, evaluations and metrics."""
        payload = PolicyScoresRequest(policy_ids=policy_ids, eval_names=eval_names, metrics=metrics)
        return await self._make_request(
            PolicyScoresData, "POST", "/scorecard/score", json=payload.model_dump(mode="json")
        )

    async def generate_heatmap_scorecard(
        self,
        training_run_ids: list[str],
        run_free_policy_ids: list[str],
        eval_names: list[str],
        metric: str,
        training_run_policy_selector: Literal["best", "latest"] = "best",
    ) -> ScorecardData:
        """Generate heatmap scorecard data based on training run and policy selection."""
        payload = ScorecardRequest(
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
            metric=metric,
            training_run_policy_selector=training_run_policy_selector,
        )
        return await self._make_request(
            ScorecardData, "POST", "/scorecard/heatmap", json=payload.model_dump(mode="json")
        )

    async def generate_training_run_scorecard(self, run_id: str, eval_names: list[str], metric: str) -> ScorecardData:
        """Generate scorecard data for a specific training run showing ALL policies."""
        payload = TrainingRunScorecardRequest(eval_names=eval_names, metric=metric)
        return await self._make_request(
            ScorecardData, "POST", f"/scorecard/training-run/{run_id}", json=payload.model_dump(mode="json")
        )

    async def generate_leaderboard_scorecard(
        self, leaderboard_id: str, selector: Literal["latest", "best"] = "latest", num_policies: int = 10
    ) -> ScorecardData:
        """Generate scorecard data for a leaderboard."""
        import uuid

        payload = LeaderboardScorecardRequest(
            leaderboard_id=uuid.UUID(leaderboard_id), selector=selector, num_policies=num_policies
        )
        return await self._make_request(
            ScorecardData, "POST", "/scorecard/leaderboard", json=payload.model_dump(mode="json")
        )

    # Stats Routes
    async def get_policy_ids(self, policy_names: list[str]) -> PolicyIdResponse:
        """Get policy IDs for given policy names."""
        query_string = "&".join([f"policy_names={name}" for name in policy_names])
        return await self._make_request(PolicyIdResponse, "GET", f"/stats/policies/ids?{query_string}")

    async def create_training_run(self, training_run_data: TrainingRunCreate) -> StatsTrainingRunResponse:
        """Create a new training run."""
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.post(
            "/stats/training-runs", headers=headers, json=training_run_data.model_dump(mode="json")
        )
        response.raise_for_status()
        data = response.json()
        return StatsTrainingRunResponse.model_validate(data)

    async def create_epoch(self, run_id: str, epoch_data: EpochCreate) -> EpochResponse:
        """Create a new policy epoch."""
        return await self._make_request(
            EpochResponse, "POST", f"/stats/training-runs/{run_id}/epochs", json=epoch_data.model_dump(mode="json")
        )

    async def create_policy(self, policy_data: PolicyCreate) -> PolicyResponse:
        """Create a new policy."""
        return await self._make_request(
            PolicyResponse, "POST", "/stats/policies", json=policy_data.model_dump(mode="json")
        )

    async def record_episode(self, episode_data: EpisodeCreate) -> EpisodeResponse:
        """Record a new episode with agent policies and metrics."""
        return await self._make_request(
            EpisodeResponse, "POST", "/stats/episodes", json=episode_data.model_dump(mode="json")
        )

    # Sweep Routes
    async def create_sweep(self, sweep_name: str, request_data: SweepCreateRequest) -> SweepCreateResponse:
        """Initialize a new sweep or return existing sweep info."""
        return await self._make_request(
            SweepCreateResponse, "POST", f"/sweeps/{sweep_name}/create_sweep", json=request_data.model_dump(mode="json")
        )

    async def get_sweep(self, sweep_name: str) -> SweepInfo:
        """Get sweep information by name."""
        return await self._make_request(SweepInfo, "GET", f"/sweeps/{sweep_name}")

    async def get_next_run_id(self, sweep_name: str) -> RunIdResponse:
        """Get the next run ID for a sweep."""
        return await self._make_request(RunIdResponse, "POST", f"/sweeps/{sweep_name}/runs/next")

    # Token Routes
    async def create_token(self, token_data: TokenCreate) -> TokenResponse:
        """Create a new machine token."""
        return await self._make_request(TokenResponse, "POST", "/tokens", json=token_data.model_dump(mode="json"))

    async def list_tokens(self) -> TokenListResponse:
        """List all machine tokens."""
        return await self._make_request(TokenListResponse, "GET", "/tokens")

    async def delete_token(self, token_id: str) -> Dict[str, str]:
        """Delete a machine token."""
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.delete(f"/tokens/{token_id}", headers=headers)
        response.raise_for_status()
        return response.json()

    async def create_cli_token(self, callback: str) -> Dict[str, Any]:
        """Create a machine token and redirect to callback URL with token parameter."""
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.get(f"/tokens/cli?callback={callback}", headers=headers)
        response.raise_for_status()
        return response.json()
