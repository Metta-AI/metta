import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from alo.assignments import build_assignments
from alo.rollouts import run_multi_episode_rollout
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult


def _run_single_simulation(
    simulation: Any,
    policy_data: Sequence[Any],
    replay_dir: str | None,
    seed: int,
    device_override: str | None = None,
) -> "SimulationRunResult":
    sim_cfg = SimulationRunConfig.model_validate(simulation)
    policy_specs = [PolicySpec.model_validate(spec) for spec in policy_data]

    proportions = list(sim_cfg.proportions) if sim_cfg.proportions is not None else [1.0] * len(policy_specs)
    if len(proportions) != len(policy_specs):
        raise ValueError("Number of proportions must match number of policies.")
    if sum(proportions) <= 0:
        raise ValueError("Total policy proportion must be positive.")
    assignments = build_assignments(sim_cfg.env.game.num_agents, proportions)
    max_action_time_ms = sim_cfg.max_action_time_ms or 10000
    rollout_result, _replay_paths = run_multi_episode_rollout(
        policy_specs=policy_specs,
        assignments=assignments,
        env_cfg=sim_cfg.env,
        episodes=sim_cfg.num_episodes,
        seed=seed,
        max_action_time_ms=max_action_time_ms,
        replay_dir=replay_dir,
        create_replay_dir=True,
        device=device_override,
    )

    return SimulationRunResult(run=sim_cfg, results=rollout_result)


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
    device_override: str | None = None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    # Sequential path for max_workers unset or 1
    if not max_workers or max_workers <= 1 or len(simulations) <= 1:
        sequential_rollouts: list[SimulationRunResult] = []
        for i, simulation in enumerate(simulations):
            on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")
            sequential_rollouts.append(
                _run_single_simulation(simulation, policy_specs, replay_dir, seed, device_override)
            )
            on_progress(f"Finished rollout for simulation {i + 1} of {len(simulations)}")

        return sequential_rollouts

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
                _run_single_simulation,
                payload,
                policy_payloads,
                os.path.join(replay_dir, f"sim_{idx}") if replay_dir else None,
                seed,
                device_override,
            ): idx
            for idx, payload in enumerate(simulation_payloads)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            simulation_rollouts[idx] = future.result()
            on_progress(f"Finished rollout for simulation {idx + 1} of {len(simulations)}")

    return simulation_rollouts
