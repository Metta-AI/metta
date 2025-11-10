import logging
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mettagrid import MettaGridEnvConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout

logger = logging.getLogger(__name__)


class SimulationRunConfig(BaseModel):
    env: MettaGridEnvConfig  # noqa: F821
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


def run_simulations(
    *,
    policy_specs: Sequence[PolicySpec],
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    on_progress: Callable[[str], None] = lambda x: None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    simulation_rollouts: list[SimulationRunResult] = []

    for i, simulation in enumerate(simulations):
        proportions = simulation.proportions

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env.game)
        multi_agent_policies: list[MultiAgentPolicy] = [
            initialize_or_load_policy(env_interface, spec) for spec in policy_specs
        ]

        on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")
        rollout_result = multi_episode_rollout(
            env_cfg=simulation.env,
            policies=multi_agent_policies,
            episodes=simulation.num_episodes,
            seed=seed,
            proportions=proportions,
            save_replay=replay_dir,
            # TODO: support this if and only if we also reflect that it happened in results
            # max_time_s=simulation.max_time_s,
            max_action_time_ms=simulation.max_action_time_ms,
        )
        on_progress(f"Finished rollout for simulation {i}")

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
            )
        )

    return simulation_rollouts
