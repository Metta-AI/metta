from pydantic import BaseModel, model_validator

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import AgentPolicy, PolicyEnvInterface, initialize_or_load_policy
from mettagrid.simulator.replay_log_writer import InMemoryReplayWriter
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import EpisodeStats
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import parse_uri, policy_spec_from_uri


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
            if not self.replay_uri.endswith(".json.z"):
                raise ValueError("Replay URI must end with .json.z")

        if any(assignment >= len(self.policy_uris) for assignment in self.assignments):
            raise ValueError("Assignment index out of range")

        if len(self.assignments) != self.env.game.num_agents:
            raise ValueError("Number of assignments must match number of agents")

        return self


class PureSingleEpisodeResult(BaseModel):
    rewards: list[float]
    action_timeouts: list[float]
    stats: EpisodeStats
    steps: int
    max_steps: int


def run_pure_single_episode(job: PureSingleEpisodeJob, device: str = "cpu") -> None:
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

    if replay_writer is not None:
        replays = replay_writer.get_completed_replays()
        if len(replays) != 1:
            raise ValueError(f"Expected 1 replay, got {len(replays)}")
        assert job.replay_uri is not None
        replay = replays[0]
        replay.write_replay(job.replay_uri)

    if job.results_uri is not None:
        results = PureSingleEpisodeResult(
            rewards=list(rollout._sim.episode_rewards),
            action_timeouts=list(rollout.timeout_counts),
            stats=rollout._sim.episode_stats,
            steps=rollout._sim.current_step,
            max_steps=rollout._sim.config.game.max_steps,
        )
        write_data(job.results_uri, results.model_dump_json(), content_type="application/json")
