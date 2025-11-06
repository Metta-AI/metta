import abc
import datetime
import logging
import typing
import uuid

import httpx
import pydantic

import metta.app_backend.clients.base_client
import metta.app_backend.routes.eval_task_routes
import metta.app_backend.routes.score_routes
import metta.app_backend.routes.sql_routes
import metta.app_backend.routes.stats_routes
import metta.common.util.collections
import metta.common.util.constants

logger = logging.getLogger("stats_client")

T = typing.TypeVar("T", bound=pydantic.BaseModel)


class StatsClient(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, backend_url: str = metta.common.util.constants.PROD_STATS_SERVER_URI, machine_token: str | None = None
    ):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any) -> None:
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> metta.app_backend.routes.stats_routes.TrainingRunResponse:
        pass

    @abc.abstractmethod
    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, typing.Any] | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpochResponse:
        pass

    @abc.abstractmethod
    def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> metta.app_backend.routes.stats_routes.PolicyResponse:
        pass

    @abc.abstractmethod
    def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        pass

    @abc.abstractmethod
    def create_task(
        self, request: metta.app_backend.routes.eval_task_routes.TaskCreateRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskResponse:
        pass

    @abc.abstractmethod
    def record_episode(
        self,
        *,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        sim_suite: str,
        env_name: str,
        stats_epoch: uuid.UUID | None = None,
        replay_url: str | None = None,
        attributes: dict[str, typing.Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpisodeResponse:
        pass

    @abc.abstractmethod
    def sql_query(self, query: str) -> metta.app_backend.routes.sql_routes.SQLQueryResponse:
        pass

    @staticmethod
    def create(stats_server_uri: typing.Optional[str]) -> "StatsClient":
        if stats_server_uri is None:
            return NoopStatsClient()

        machine_token = metta.app_backend.clients.base_client.get_machine_token(stats_server_uri)
        if machine_token is None:
            raise metta.app_backend.clients.base_client.NotAuthenticatedError(
                f"No machine token found for {stats_server_uri}"
            )
        stats_client = HttpStatsClient(backend_url=stats_server_uri, machine_token=machine_token)
        stats_client._validate_authenticated()
        return stats_client


# TODO: REMOVE THIS
class NoopStatsClient(StatsClient):
    def __init__(self):
        self.id = uuid.uuid1()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any) -> None:
        pass

    def close(self):
        pass

    def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> metta.app_backend.routes.stats_routes.TrainingRunResponse:
        return metta.app_backend.routes.stats_routes.TrainingRunResponse(id=self.id)

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, typing.Any] | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpochResponse:
        return metta.app_backend.routes.stats_routes.EpochResponse(id=self.id)

    def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        pass

    def create_task(
        self, request: metta.app_backend.routes.eval_task_routes.TaskCreateRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskResponse:
        return metta.app_backend.routes.eval_task_routes.TaskResponse(
            id=self.id,
            policy_id=uuid.uuid4(),
            sim_suite="default_suite",
            status="unprocessed",
            created_at=datetime.datetime.now(),
            attributes={},
            retries=0,
            updated_at=datetime.datetime.now(),
        )

    def record_episode(
        self,
        *,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        sim_suite: str,
        env_name: str,
        stats_epoch: uuid.UUID | None = None,
        replay_url: str | None = None,
        attributes: dict[str, typing.Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpisodeResponse:
        return metta.app_backend.routes.stats_routes.EpisodeResponse(id=self.id)

    def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> metta.app_backend.routes.stats_routes.PolicyResponse:
        return metta.app_backend.routes.stats_routes.PolicyResponse(id=self.id)

    def sql_query(self, query: str) -> metta.app_backend.routes.sql_routes.SQLQueryResponse:
        return metta.app_backend.routes.sql_routes.SQLQueryResponse(columns=[], rows=[], row_count=0)


class HttpStatsClient(StatsClient):
    """Synchronous wrapper around AsyncStatsClient using httpx sync client."""

    def __init__(
        self, backend_url: str = metta.common.util.constants.PROD_STATS_SERVER_URI, machine_token: str | None = None
    ):
        self._backend_url = backend_url
        self._http_client = httpx.Client(
            base_url=backend_url,
            timeout=30.0,
        )

        self.machine_token = machine_token or metta.app_backend.clients.base_client.get_machine_token(backend_url)
        self._machine_token = self.machine_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any) -> None:
        self.close()

    def close(self):
        self._http_client.close()

    def _make_sync_request(self, response_type: typing.Type[T], method: str, url: str, **kwargs) -> T:
        headers = metta.common.util.collections.remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    def _validate_authenticated(self) -> str:
        import metta.app_backend.server

        auth_user = self._make_sync_request(metta.app_backend.server.WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise metta.app_backend.clients.base_client.NotAuthenticatedError(
                f"Not authenticated. User: {auth_user.user_email}"
            )
        return auth_user.user_email

    def get_policy_ids(self, policy_names: list[str]) -> metta.app_backend.routes.stats_routes.PolicyIdResponse:
        return self._make_sync_request(
            metta.app_backend.routes.stats_routes.PolicyIdResponse,
            "GET",
            "/stats/policies/ids",
            params={"policy_names": policy_names},
        )

    def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> metta.app_backend.routes.stats_routes.TrainingRunResponse:
        data = metta.app_backend.routes.stats_routes.TrainingRunCreate(
            name=name,
            attributes=attributes or {},
            url=url,
            description=description,
            tags=tags,
        )
        return self._make_sync_request(
            metta.app_backend.routes.stats_routes.TrainingRunResponse,
            "POST",
            "/stats/training-runs",
            json=data.model_dump(mode="json"),
        )

    def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        headers = metta.common.util.collections.remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(
            "PATCH", f"/stats/training-runs/{run_id}/status", headers=headers, json={"status": status}
        )
        response.raise_for_status()

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, typing.Any] | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpochResponse:
        data = metta.app_backend.routes.stats_routes.EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        return self._make_sync_request(
            metta.app_backend.routes.stats_routes.EpochResponse,
            "POST",
            f"/stats/training-runs/{run_id}/epochs",
            json=data.model_dump(mode="json"),
        )

    def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> metta.app_backend.routes.stats_routes.PolicyResponse:
        data = metta.app_backend.routes.stats_routes.PolicyCreate(
            name=name,
            description=description,
            url=url,
            epoch_id=epoch_id,
        )
        return self._make_sync_request(
            metta.app_backend.routes.stats_routes.PolicyResponse,
            "POST",
            "/stats/policies",
            json=data.model_dump(mode="json"),
        )

    def record_episode(
        self,
        *,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        sim_suite: str,
        env_name: str,
        stats_epoch: uuid.UUID | None = None,
        replay_url: str | None = None,
        attributes: dict[str, typing.Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> metta.app_backend.routes.stats_routes.EpisodeResponse:
        data = metta.app_backend.routes.stats_routes.EpisodeCreate(
            agent_policies=agent_policies,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id,
            stats_epoch=stats_epoch,
            sim_suite=sim_suite,
            env_name=env_name,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=eval_task_id,
            tags=tags,
            thumbnail_url=thumbnail_url,
        )
        return self._make_sync_request(
            metta.app_backend.routes.stats_routes.EpisodeResponse,
            "POST",
            "/stats/episodes",
            json=data.model_dump(mode="json"),
        )

    def create_task(
        self, request: metta.app_backend.routes.eval_task_routes.TaskCreateRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskResponse:
        return self._make_sync_request(
            metta.app_backend.routes.eval_task_routes.TaskResponse,
            "POST",
            "/tasks",
            json=request.model_dump(mode="json"),
        )

    def get_all_tasks(
        self, filters: metta.app_backend.routes.eval_task_routes.TaskFilterParams | None = None
    ) -> metta.app_backend.routes.eval_task_routes.TasksResponse:
        params = filters.model_dump(mode="json", exclude_none=True) if filters else {}
        return self._make_sync_request(
            metta.app_backend.routes.eval_task_routes.TasksResponse, "GET", "/tasks/all", params=params
        )

    def get_policy_scores(
        self, request: metta.app_backend.routes.score_routes.PolicyScoresRequest
    ) -> metta.app_backend.routes.score_routes.PolicyScoresData:
        return self._make_sync_request(
            metta.app_backend.routes.score_routes.PolicyScoresData,
            "POST",
            "/scorecard/score",
            json=request.model_dump(mode="json"),
        )

    def sql_query(self, query: str) -> metta.app_backend.routes.sql_routes.SQLQueryResponse:
        return self._make_sync_request(
            metta.app_backend.routes.sql_routes.SQLQueryResponse, "POST", "/sql/query", json={"query": query}
        )
