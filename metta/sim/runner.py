import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.doxascope.doxascope_data import DoxascopeEventHandler, DoxascopeLogger
from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import SimulatorEventHandler
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout

logger = logging.getLogger(__name__)


def _run_single_simulation(
    simulation: Any,
    policy_data: Sequence[Any],
    replay_dir: str | None,
    seed: int,
    device_override: str | None = None,
) -> "SimulationRunResult":
    sim_cfg = SimulationRunConfig.model_validate(simulation)
    policy_specs = [PolicySpec.model_validate(spec) for spec in policy_data]

    env_interface = PolicyEnvInterface.from_mg_cfg(sim_cfg.env)
    multi_agent_policies: list[MultiAgentPolicy] = [
        initialize_or_load_policy(env_interface, spec, device_override) for spec in policy_specs
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
    device_override: str | None = None,
    event_handlers: list[SimulatorEventHandler] | None = None,
) -> list[SimulationRunResult]:
    if not policy_specs:
        raise ValueError("At least one policy spec is required")

    # Helper to check if DoxascopeEventHandler exists in event_handlers
    def has_doxascope_handler(handlers: list[SimulatorEventHandler] | None) -> bool:
        if handlers is None:
            return False
        return any(isinstance(h, DoxascopeEventHandler) for h in handlers)

    # Doxascope requires sequential processing because:
    # 1. The logger tracks state across timesteps within the same process
    # 2. ProcessPoolExecutor spawns separate processes that can't share logger state
    # 3. doxascope_enabled simulations need access to the configured logger
    uses_doxascope = has_doxascope_handler(event_handlers) or any(sim.doxascope_enabled for sim in simulations)

    # Sequential path: for doxascope, max_workers unset/1, or single simulation
    if uses_doxascope or not max_workers or max_workers <= 1 or len(simulations) <= 1:
        simulation_rollouts: list[SimulationRunResult] = []

        for i, simulation in enumerate(simulations):
            # Clone event handlers for this simulation
            # If there's a DoxascopeEventHandler, clone its logger and create a new handler
            current_handlers: list[SimulatorEventHandler] = []
            current_logger: DoxascopeLogger | None = None

            if event_handlers:
                for handler in event_handlers:
                    if isinstance(handler, DoxascopeEventHandler):
                        # Clone the logger with a new simulation ID
                        prefix = simulation.episode_tags.get("name", "eval")
                        sim_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
                        current_logger = handler._logger.clone(sim_id)
                        # Create new event handler with cloned logger
                        current_handlers.append(DoxascopeEventHandler(current_logger))
                    else:
                        # Keep other handlers as-is
                        current_handlers.append(handler)
            elif simulation.doxascope_enabled:
                # Create a new logger if simulation requires doxascope but none was provided
                prefix = simulation.episode_tags.get("name", "eval")
                simulation_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
                current_logger = DoxascopeLogger(enabled=True, simulation_id=simulation_id)
                current_handlers.append(DoxascopeEventHandler(current_logger))

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
                proportions=simulation.proportions,
                save_replay=replay_dir,
                max_action_time_ms=simulation.max_action_time_ms,
                event_handlers=current_handlers if current_handlers else None,
            )

            if current_logger:
                current_logger.save()

            on_progress(f"Finished rollout for simulation {i + 1} of {len(simulations)}")

            simulation_rollouts.append(
                SimulationRunResult(
                    run=simulation,
                    results=rollout_result,
                )
            )

        return simulation_rollouts

    # Parallel path (no doxascope support in parallel mode)
    simulation_rollouts = [None] * len(simulations)  # type: ignore[assignment]

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
