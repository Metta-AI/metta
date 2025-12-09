import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.doxascope.doxascope_data import DoxascopeLogger
from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout


def _run_single_simulation(
    simulation: Any,
    policy_data: Sequence[Any],
    replay_dir: str | None,
    seed: int,
) -> "SimulationRunResult":
    sim_cfg = SimulationRunConfig.model_validate(simulation)
    policy_specs = [PolicySpec.model_validate(spec) for spec in policy_data]

    env_interface = PolicyEnvInterface.from_mg_cfg(sim_cfg.env)
    multi_agent_policies: list[MultiAgentPolicy] = [
        initialize_or_load_policy(env_interface, spec) for spec in policy_specs
    ]

    if replay_dir:
        os.makedirs(replay_dir, exist_ok=True)

    rollout_result = multi_episode_rollout(
        env_cfg=sim_cfg.env,
        policies=multi_agent_policies,
        episodes=sim_cfg.num_episodes,
        seed=seed,
        proportions=sim_cfg.proportions,
        save_replay=replay_dir,
        max_action_time_ms=sim_cfg.max_action_time_ms,
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
    max_workers: int | None = None,
    on_progress: Callable[[str], None] = lambda x: None,
    doxascope_logger: DoxascopeLogger | None = None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    # Check if any simulation has doxascope enabled
    any_doxascope = doxascope_logger is not None or any(sim.doxascope_enabled for sim in simulations)

    # Sequential path for max_workers unset/1, single simulation, or when doxascope is enabled
    # (doxascope logger can't be pickled for multiprocessing)
    if not max_workers or max_workers <= 1 or len(simulations) <= 1 or any_doxascope:
        sequential_rollouts: list[SimulationRunResult] = []
        for i, simulation in enumerate(simulations):
            on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")

            # Set up doxascope logger for this simulation
            current_logger = None
            if doxascope_logger is not None:
                prefix = simulation.episode_tags.get("name", "eval")
                sim_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
                current_logger = doxascope_logger.clone(sim_id)
            elif simulation.doxascope_enabled:
                prefix = simulation.episode_tags.get("name", "eval")
                simulation_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
                current_logger = DoxascopeLogger(enabled=True, simulation_id=simulation_id)

            env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
            multi_agent_policies: list[MultiAgentPolicy] = [
                initialize_or_load_policy(env_interface, spec) for spec in policy_specs
            ]

            rollout_result = multi_episode_rollout(
                env_cfg=simulation.env,
                policies=multi_agent_policies,
                episodes=simulation.num_episodes,
                seed=seed,
                proportions=simulation.proportions,
                save_replay=replay_dir,
                max_action_time_ms=simulation.max_action_time_ms,
                doxascope_logger=current_logger,
            )

            if current_logger:
                current_logger.save()

            sequential_rollouts.append(SimulationRunResult(run=simulation, results=rollout_result))
            on_progress(f"Finished rollout for simulation {i + 1} of {len(simulations)}")

        return sequential_rollouts

    # Parallel path (no doxascope support - can't pickle the logger)
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
            ): idx
            for idx, payload in enumerate(simulation_payloads)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            simulation_rollouts[idx] = future.result()
            on_progress(f"Finished rollout for simulation {idx + 1} of {len(simulations)}")

    return simulation_rollouts
