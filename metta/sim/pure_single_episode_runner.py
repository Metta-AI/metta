import json
import socket
import sys
from contextlib import contextmanager, nullcontext

from pydantic import BaseModel, model_validator

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import AgentPolicy, PolicyEnvInterface, initialize_or_load_policy
from mettagrid.policy.prepare_policy_spec import download_policy_spec_from_s3
from mettagrid.simulator.replay_log_writer import EpisodeReplay, InMemoryReplayWriter
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import EpisodeStats
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import parse_uri, policy_spec_from_uri, resolve_uri


class PureSingleEpisodeJob(BaseModel):
    policy_uris: list[str]

    # It is important that this is explicit, else the results will have to include the choices we made
    # when randomizing
    assignments: list[int]

    env: MettaGridConfig

    # For now, this only supports file:// scheme. Will eventually support https:// to send to s3
    results_uri: str | None  # Contains EpisodeRolloutResult
    replay_uri: str | None  # Where to place replay file. If missing, do not generate a replay

    # There's no way to ask us to generate a seed; the caller has to pick one
    seed: int = 0

    max_action_time_ms: int = 10000

    @model_validator(mode="after")
    def validate_replay_uri(self) -> "PureSingleEpisodeJob":
        # replay_uri and results_uri are both file:// URIs and point to local directories
        for uri in [self.replay_uri, self.results_uri]:
            if uri is None:
                continue

            parsed = parse_uri(uri, allow_none=False)
            if parsed.scheme != "file":
                raise ValueError(f"URI {uri} must be a file:// URI")

            if not parsed.local_path.parent.exists():
                raise ValueError(f"Directory {parsed.local_path.parent} does not exist")

        if self.replay_uri is not None:
            if self.replay_uri.endswith(".json.z"):
                pass
            elif self.replay_uri.endswith(".json.gz"):
                pass
            else:
                raise ValueError("Replay URI must end with .json.z or .json.gz")

        if not all(0 <= assignment < len(self.policy_uris) for assignment in self.assignments):
            raise ValueError("Assignment index out of range")

        if len(self.assignments) != self.env.game.num_agents:
            raise ValueError("Number of assignments must match number of agents")

        return self


class PureSingleEpisodeResult(BaseModel):
    rewards: list[float]
    action_timeouts: list[int]
    stats: EpisodeStats
    steps: int


@contextmanager
def _no_python_sockets():
    _real_socket = socket.socket
    _real_getaddrinfo = socket.getaddrinfo

    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access disabled")

    socket.socket = _blocked
    socket.getaddrinfo = _blocked

    try:
        yield
    finally:
        socket.socket = _real_socket
        socket.getaddrinfo = _real_getaddrinfo


def run_single_episode(job: PureSingleEpisodeJob, allow_network: bool = False, device: str = "cpu") -> None:
    # Pull each policy onto the local filesystem, leave them as zip files
    local_uris: list[str] = []
    for uri in job.policy_uris:
        resolved = resolve_uri(uri)
        local: str | None = None
        if resolved.scheme == "file":
            local = resolved.local_path.as_uri()
        elif resolved.scheme == "s3":
            local = download_policy_spec_from_s3(resolved.canonical, remove_downloaded_copy_on_exit=True).as_uri()
        if local is None:
            raise RuntimeError(f"could not resolve policy {uri}")
        local_uris.append(local)
    job.policy_uris = local_uris
    with (_no_python_sockets if not allow_network else nullcontext)():
        results, replay = run_pure_single_episode(job, device)

    if job.replay_uri is not None:
        if replay is not None:
            if job.replay_uri.endswith(".z"):
                replay.set_compression("zlib")
            elif job.replay_uri.endswith(".gz"):
                replay.set_compression("gzip")
            replay.write_replay(job.replay_uri)
        else:
            raise ValueError("No replay was generated")
    if job.results_uri is not None:
        write_data(job.results_uri, results.model_dump_json(), content_type="application/json")


def run_pure_single_episode(
    job: PureSingleEpisodeJob,
    device: str,
) -> tuple[PureSingleEpisodeResult, EpisodeReplay | None]:
    policy_specs = [policy_spec_from_uri(uri) for uri in job.policy_uris]

    env_interface = PolicyEnvInterface.from_mg_cfg(job.env)
    agent_policies: list[AgentPolicy] = [
        initialize_or_load_policy(env_interface, policy_specs[assignment], device_override=device).agent_policy(
            agent_id
        )
        for agent_id, assignment in enumerate(job.assignments)
    ]
    replay_writer: InMemoryReplayWriter | None = None
    if job.replay_uri is not None:
        replay_writer = InMemoryReplayWriter()

    rollout = Rollout(
        job.env,
        agent_policies,
        max_action_time_ms=job.max_action_time_ms,
        render_mode="none",
        seed=job.seed,
        event_handlers=[replay_writer] if replay_writer is not None else None,
    )
    rollout.run_until_done()

    results = PureSingleEpisodeResult(
        rewards=list(rollout._sim.episode_rewards),
        action_timeouts=list(rollout.timeout_counts),
        stats=rollout._sim.episode_stats,
        steps=rollout._sim.current_step,
    )
    replay: EpisodeReplay | None = None
    if replay_writer is not None:
        replays = replay_writer.get_completed_replays()
        if len(replays) != 1:
            raise ValueError(f"Expected 1 replay, got {len(replays)}")
        assert job.replay_uri is not None
        replay = replays[0]

    return results, replay


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    job = PureSingleEpisodeJob.model_validate(args["job"])
    device = args["device"]
    allow_network = args.get("allow_network", False)
    run_single_episode(job, allow_network=allow_network, device=device)
