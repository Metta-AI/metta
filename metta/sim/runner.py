import logging
import uuid
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.doxascope.doxascope_data import DoxascopeLogger
from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout

logger = logging.getLogger(__name__)


class SimulationRunConfig(BaseModel):
    env: MettaGridConfig  # noqa: F821
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    proportions: Sequence[float] | None = None

    max_action_time_ms: int | None = Field(
        default=10000, description="Maximum time (in ms) a policy is given to take an action"
    )
    episode_tags: dict[str, str] = Field(default_factory=dict, description="Tags to add to each episode")
    doxascope_enabled: bool = Field(default=False, description="Enable Doxascope logger")


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
    doxascope_logger: DoxascopeLogger | None = None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    simulation_rollouts: list[SimulationRunResult] = []

    for i, simulation in enumerate(simulations):
        proportions = simulation.proportions

        if doxascope_logger is None and simulation.doxascope_enabled:
            prefix = simulation.episode_tags.get("name", "eval")
            simulation_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
            current_logger = DoxascopeLogger(enabled=True, simulation_id=simulation_id)
        elif doxascope_logger:
            prefix = simulation.episode_tags.get("name", "eval")
            sim_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
            current_logger = doxascope_logger.clone(sim_id)
        else:
            current_logger = None

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
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
            doxascope_logger=current_logger,
        )

        if current_logger:
            current_logger.save()

        on_progress(f"Finished rollout for simulation {i}")

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
            )
        )

    return simulation_rollouts
