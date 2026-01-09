import http
import logging
import socket
from typing import Annotated

import fastapi
import typer
import uvicorn
from google.protobuf import json_format

from metta.common.util.log_config import init_logging, suppress_noisy_logs
from metta.protobuf.sim.policy_v1 import policy_pb2
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy

logger = logging.getLogger(__name__)

cli = typer.Typer()


class EpisodeNotFoundError(Exception):
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        super().__init__(f"unknown episode_id: {episode_id}")


class AgentNotFoundError(Exception):
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        super().__init__(f"unknown agent_id: {agent_id}")


class Episode:
    def __init__(self, episode_id: str, policy: MultiAgentPolicy, agent_ids: list[int]):
        self.episode_id = episode_id
        self.policy = policy
        self.agent_ids = agent_ids
        self.agent_policies: dict[int, AgentPolicy] = {}

    def create_agent_policies(self):
        for agent_id in self.agent_ids:
            self.agent_policies[agent_id] = self.policy.agent_policy(agent_id)


class PolicyService:
    def __init__(self, policy: MultiAgentPolicy):
        self.policy = policy
        self._episodes: dict[str, Episode] = {}

    def prepare_policy(self, req: policy_pb2.PreparePolicyRequest) -> policy_pb2.PreparePolicyResponse:
        logger.info("PreparePolicy: %s", req)
        episode = Episode(
            episode_id=req.episode_id,
            policy=self.policy,
            agent_ids=list(req.agent_ids),
        )
        episode.create_agent_policies()
        self._episodes[req.episode_id] = episode
        return policy_pb2.PreparePolicyResponse()

    def batch_step(self, req: policy_pb2.BatchStepRequest) -> policy_pb2.BatchStepResponse:
        logger.info("BatchStep: %s", req)
        episode = self._episodes.get(req.episode_id)
        if episode is None:
            raise EpisodeNotFoundError(req.episode_id)

        resp = policy_pb2.BatchStepResponse()
        for agent_obs in req.agent_observations:
            agent_id = agent_obs.agent_id
            agent_policy = episode.agent_policies.get(agent_id)
            if agent_policy is None:
                raise AgentNotFoundError(agent_id)
            # TODO: construct proper AgentObservation from agent_obs.observations
            action = agent_policy.step(None)  # type: ignore[arg-type]
            # TODO: proper action_id lookup; for now parse from name
            action_id = int(action.name)
            resp.agent_actions.append(policy_pb2.AgentActions(agent_id=agent_id, action_id=[action_id]))
        return resp


def create_app(service: PolicyService, *, verbose: bool = False) -> fastapi.FastAPI:
    app = fastapi.FastAPI()

    @app.exception_handler(EpisodeNotFoundError)
    async def handle_episode_not_found(request: fastapi.Request, exc: EpisodeNotFoundError):
        return fastapi.responses.JSONResponse(
            status_code=http.HTTPStatus.NOT_FOUND,
            content={"detail": "episode not found"},
        )

    @app.exception_handler(AgentNotFoundError)
    async def handle_agent_not_found(request: fastapi.Request, exc: AgentNotFoundError):
        return fastapi.responses.JSONResponse(
            status_code=http.HTTPStatus.NOT_FOUND,
            content={"detail": "agent not found"},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # TODO: factor out the repeated handler pattern (parse proto request, call service,
    # serialize proto response) into a generic wrapper or decorator.

    @app.post("/metta.protobuf.sim.policy_v1.Policy/PreparePolicy")
    async def prepare_policy(request: fastapi.Request):
        body = await request.body()
        try:
            req = json_format.Parse(body, policy_pb2.PreparePolicyRequest())
        except json_format.ParseError as e:
            if verbose:
                logger.warning("PreparePolicy: invalid request: %s", e)
            raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="invalid request body") from e
        resp = service.prepare_policy(req)
        return fastapi.Response(
            content=json_format.MessageToJson(resp),
            media_type="application/json",
        )

    @app.post("/metta.protobuf.sim.policy_v1.Policy/BatchStep")
    async def batch_step(request: fastapi.Request):
        body = await request.body()
        try:
            req = json_format.Parse(body, policy_pb2.BatchStepRequest())
        except json_format.ParseError as e:
            if verbose:
                logger.warning("BatchStep: invalid request: %s", e)
            raise fastapi.HTTPException(status_code=http.HTTPStatus.BAD_REQUEST, detail="invalid request body") from e
        resp = service.batch_step(req)
        return fastapi.Response(
            content=json_format.MessageToJson(resp),
            media_type="application/json",
        )

    return app


@cli.command()
def main(
    policy: Annotated[str, typer.Option(help="Policy ID")],
    host: Annotated[str, typer.Option(help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind to (0 for OS-assigned)")] = 8000,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """Serve a policy over HTTP using the Policy protocol (JSON)."""
    # TODO: load MultiAgentPolicy from policy URI
    service = PolicyService(None)  # type: ignore[arg-type]
    app = create_app(service, verbose=verbose)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]
    logger.info("Serving policy %s on %s:%d", policy, host, actual_port)

    config = uvicorn.Config(app, fd=sock.fileno())
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    cli()
