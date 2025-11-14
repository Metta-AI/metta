import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout
from mettagrid.simulator.replay_log_writer import ReplayLogWriter


class SimulationRunConfig(BaseModel):
    env: MettaGridConfig  # noqa: F821
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    proportions: Sequence[float] | None = None

    max_action_time_ms: int | None = Field(
        default=10000, description="Maximum time (in ms) a policy is given to take an action"
    )
    episode_tags: dict[str, str] = Field(default_factory=dict, description="Tags to add to each episode")


class SimulationRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: SimulationRunConfig
    results: MultiEpisodeRolloutResult
    replay_urls: dict[str, str]


MultiAgentPolicyInitializer = Callable[[PolicyEnvInterface], MultiAgentPolicy]


def _make_replay_writer(replay_dir: str | None) -> ReplayLogWriter | None:
    if replay_dir is None:
        return None
    replay_root = Path(replay_dir).expanduser()
    unique_dir = replay_root / uuid.uuid4().hex[:12]
    unique_dir.mkdir(parents=True, exist_ok=True)
    return ReplayLogWriter(str(unique_dir))


def run_simulations(
    *,
    policy_specs: Sequence[PolicySpec] | None = None,
    policy_initializers: Sequence[MultiAgentPolicyInitializer] | None = None,
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    enable_replays: bool = True,
) -> list[SimulationRunResult]:
    if policy_initializers is None:
        if policy_specs is None:
            msg = "Either policy_specs or policy_initializers must be provided"
            raise ValueError(msg)
        policy_initializers = [partial(initialize_or_load_policy, policy_spec=spec) for spec in policy_specs]
    elif policy_specs is not None:
        msg = "Provide only policy_specs or policy_initializers, not both"
        raise ValueError(msg)

    if not policy_initializers:
        msg = "At least one policy initializer is required"
        raise ValueError(msg)

    simulation_rollouts: list[SimulationRunResult] = []

    for simulation in simulations:
        replay_writer: ReplayLogWriter | None = None
        if enable_replays and replay_dir:
            replay_writer = _make_replay_writer(replay_dir)

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = [pi(env_interface) for pi in policy_initializers]

        rollout_result = multi_episode_rollout(
            env_cfg=simulation.env,
            policies=multi_agent_policies,
            episodes=simulation.num_episodes,
            seed=seed,
            proportions=simulation.proportions,
            # TODO: support this if and only if we also reflect that it happened in results
            # max_time_s=simulation.max_time_s,
            max_action_time_ms=simulation.max_action_time_ms,
            event_handlers=[replay_writer] if replay_writer else None,
        )

        replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
                replay_urls=replay_urls,
            )
        )

    return simulation_rollouts
