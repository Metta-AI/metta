import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout

logger = logging.getLogger(__name__)


def _run_single_simulation(
    *,
    sim_idx: int,
    simulation: "SimulationRunConfig",
    policy_specs: Sequence[PolicySpec],
    replay_dir: str | None,
    seed: int,
) -> "SimulationRunResult":
    env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
    multi_agent_policies: list[MultiAgentPolicy] = [
        initialize_or_load_policy(env_interface, spec) for spec in policy_specs
    ]

    if replay_dir:
        os.makedirs(replay_dir, exist_ok=True)

    rollout_result = multi_episode_rollout(
        env_cfg=simulation.env,
        policies=multi_agent_policies,
        episodes=simulation.num_episodes,
        seed=seed + sim_idx,
        proportions=simulation.proportions,
        save_replay=replay_dir,
        max_action_time_ms=simulation.max_action_time_ms,
    )

    return SimulationRunResult(run=simulation, results=rollout_result)


def _run_single_simulation_from_payload(
    sim_idx: int,
    sim_payload: dict,
    policy_payloads: list[dict],
    replay_dir: str | None,
    seed: int,
) -> "SimulationRunResult":
    """Adapter for ProcessPool workers that rebuilds Pydantic models."""

    simulation = SimulationRunConfig.model_validate(sim_payload)
    policy_specs = [PolicySpec.model_validate(spec_payload) for spec_payload in policy_payloads]
    per_sim_replay_dir = None
    if replay_dir:
        per_sim_replay_dir = os.path.join(replay_dir, f"sim_{sim_idx}")
    return _run_single_simulation(
        sim_idx=sim_idx,
        simulation=simulation,
        policy_specs=policy_specs,
        replay_dir=per_sim_replay_dir,
        seed=seed,
    )


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


def run_simulations(
    *,
    policy_specs: Sequence[PolicySpec],
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    max_workers: int | None = None,
    on_progress: Callable[[str], None] = lambda x: None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    materialized_specs = list(policy_specs)

    # Sequential path for max_workers unset or 1
    if not max_workers or max_workers <= 1:
        simulation_rollouts: list[SimulationRunResult] = []
        for i, simulation in enumerate(simulations):
            on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")
            simulation_rollouts.append(
                _run_single_simulation(
                    sim_idx=i,
                    simulation=simulation,
                    policy_specs=materialized_specs,
                    replay_dir=replay_dir,
                    seed=seed,
                )
            )
            on_progress(f"Finished rollout for simulation {i + 1} of {len(simulations)}")

        return simulation_rollouts

    # Parallel path
    simulation_rollouts: list[SimulationRunResult] = [None] * len(simulations)  # type: ignore[assignment]

    # Serialize configs to avoid pickling issues
    simulation_payloads = [sim.model_dump(mode="json") for sim in simulations]
    policy_payloads = [spec.model_dump(mode="json") for spec in policy_specs]

    on_progress(f"Launching {len(simulations)} eval rollouts with up to {max_workers} workers")

    # Use spawn in parallel mode; it's safer for CUDA and avoids subtle fork issues.
    mp_context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        future_to_idx = {
            executor.submit(
                _run_single_simulation_from_payload,
                idx,
                payload,
                policy_payloads,
                replay_dir,
                seed,
            ): idx
            for idx, payload in enumerate(simulation_payloads)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            simulation_rollouts[idx] = future.result()
            on_progress(f"Finished rollout for simulation {idx + 1} of {len(simulations)}")

    return simulation_rollouts
