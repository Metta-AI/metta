from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import MultiAgentPolicy
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


def run_simulations(
    policy_initializers: Sequence[MultiAgentPolicyInitializer],
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    enable_replays: bool = True,
    on_progress: Callable[[str], None] = lambda x: None,
) -> list[SimulationRunResult]:
    simulation_rollouts: list[SimulationRunResult] = []

    for i, simulation in enumerate(simulations):
        proportions = simulation.proportions
        replay_writer: ReplayLogWriter | None = None
        if enable_replays and replay_dir:
            replay_writer = ReplayLogWriter(str(replay_dir))

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = [pi(env_interface) for pi in policy_initializers]

        on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")
        rollout_result = multi_episode_rollout(
            env_cfg=simulation.env,
            policies=multi_agent_policies,
            episodes=simulation.num_episodes,
            seed=seed,
            proportions=proportions,
            # TODO: support this if and only if we also reflect that it happened in results
            # max_time_s=simulation.max_time_s,
            max_action_time_ms=simulation.max_action_time_ms,
            event_handlers=[replay_writer] if replay_writer else None,
        )
        on_progress(f"Finished rollout for simulation {simulation.env.name}")

        replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
                replay_urls=replay_urls,
            )
        )

    return simulation_rollouts
