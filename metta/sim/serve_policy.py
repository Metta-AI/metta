import http
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import fastapi
import typer
import uvicorn
from google.protobuf import json_format

from metta.common.util.log_config import init_logging, suppress_noisy_logs
from metta.protobuf.sim.policy_v1 import policy_pb2
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

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


class UnsupportedObservationFormatError(Exception):
    def __init__(self, format: int):
        self.format = format
        super().__init__(f"unsupported observation format: {format}")


def parse_triplet_v1(data: bytes, features: dict[int, ObservationFeatureSpec]) -> list[ObservationToken]:
    """Parse TRIPLET_V1 format observations into ObservationTokens.

    Every 3 bytes: [location_byte, feature_id, value]
    - location_byte: row = (b >> 4) & 0x0F, col = b & 0x0F; 0xFF means skip
    """
    tokens = []
    for i in range(0, len(data), 3):
        if i + 2 >= len(data):
            break
        loc_byte, feature_id, value = data[i], data[i + 1], data[i + 2]
        if loc_byte == 0xFF:
            continue
        row = (loc_byte >> 4) & 0x0F
        col = loc_byte & 0x0F
        feature = features.get(feature_id)
        if feature is None:
            continue
        tokens.append(
            ObservationToken(
                feature=feature,
                location=(col, row),
                value=value,
                raw_token=(loc_byte, feature_id, value),
            )
        )
    return tokens


ObservationParser = Callable[[bytes, dict[int, ObservationFeatureSpec]], list[ObservationToken]]

OBSERVATION_PARSERS: dict[int, ObservationParser] = {
    policy_pb2.AgentObservations.Format.TRIPLET_V1: parse_triplet_v1,
}


@dataclass
class Episode:
    episode_id: str
    policy: MultiAgentPolicy
    features: dict[int, ObservationFeatureSpec]
    actions: dict[str, int]
    parse_observations: ObservationParser
    agent_policies: dict[int, AgentPolicy]


EnvInterfaceAdapter = Callable[[policy_pb2.PreparePolicyRequest], PolicyEnvInterface]


class PolicyService:
    def __init__(
        self,
        policy_factory: Callable[[PolicyEnvInterface], MultiAgentPolicy],
        env_interface_adapter: EnvInterfaceAdapter,
    ):
        self._policy_factory = policy_factory
        self._env_interface_adapter = env_interface_adapter
        self._episodes: dict[str, Episode] = {}

    def prepare_policy(self, req: policy_pb2.PreparePolicyRequest) -> policy_pb2.PreparePolicyResponse:
        logger.info("PreparePolicy: %s", req)
        parse_observations = OBSERVATION_PARSERS.get(req.observations_format)
        if parse_observations is None:
            raise UnsupportedObservationFormatError(req.observations_format)
        policy_env = self._env_interface_adapter(req)
        policy = self._policy_factory(policy_env)
        features = {
            f.id: ObservationFeatureSpec(id=f.id, name=f.name, normalization=f.normalization)
            for f in req.game_rules.features
        }
        actions = {a.name: a.id for a in req.game_rules.actions}
        agent_policies = {agent_id: policy.agent_policy(agent_id) for agent_id in req.agent_ids}
        episode = Episode(
            episode_id=req.episode_id,
            policy=policy,
            features=features,
            actions=actions,
            parse_observations=parse_observations,
            agent_policies=agent_policies,
        )
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
            tokens = episode.parse_observations(agent_obs.observations, episode.features)
            observation = AgentObservation(agent_id=agent_id, tokens=tokens)
            action = agent_policy.step(observation)
            action_id = episode.actions.get(action.name)
            actions: list[int] = []
            if action_id is None:
                logger.warning("episode %r agent %d returned unknown action %r", req.episode_id, agent_id, action.name)
            else:
                actions.append(action_id)
            resp.agent_actions.append(policy_pb2.AgentActions(agent_id=agent_id, action_id=actions))
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

    @app.exception_handler(UnsupportedObservationFormatError)
    async def handle_unsupported_format(request: fastapi.Request, exc: UnsupportedObservationFormatError):
        return fastapi.responses.JSONResponse(
            status_code=http.HTTPStatus.BAD_REQUEST,
            content={"detail": "unsupported observation format"},
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
    env_interface_file: Annotated[Path, typer.Option(help="Path to PolicyEnvInterface JSON file")],
    host: Annotated[str, typer.Option(help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind to (0 for OS-assigned)")] = 8000,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging")] = False,
):
    """Serve a policy over HTTP using the Policy protocol (JSON)."""
    env_interface = PolicyEnvInterface.model_validate_json(env_interface_file.read_text())
    policy_spec = policy_spec_from_uri(policy)

    def policy_factory(env: PolicyEnvInterface) -> MultiAgentPolicy:
        return initialize_or_load_policy(env, policy_spec, device_override="cpu")

    def env_interface_adapter(_req: policy_pb2.PreparePolicyRequest) -> PolicyEnvInterface:
        # NOTE: env_interface from file may not match what PreparePolicy provides.
        # Mismatches in num_agents or available actions can cause runtime errors.
        # Future work: construct PolicyEnvInterface from PreparePolicyRequest.
        return env_interface

    service = PolicyService(policy_factory, env_interface_adapter)
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
