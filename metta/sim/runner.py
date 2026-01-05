import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field

from metta.rl.slot import PolicySlotConfig, SlotControllerPolicy, SlotRegistry
from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout


class SimulationRunConfig(BaseModel):
    env: MettaGridConfig  # noqa: F821
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    proportions: Sequence[float] | None = None
    policy_slots: Sequence[PolicySlotConfig] | None = None
    agent_slot_map: Sequence[str] | None = None

    max_action_time_ms: int | None = Field(
        default=10000, description="Maximum time (in ms) a policy is given to take an action"
    )
    episode_tags: dict[str, str] = Field(default_factory=dict, description="Tags to add to each episode")


class SimulationRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: SimulationRunConfig
    results: MultiEpisodeRolloutResult
    per_slot_returns: dict[str, float] | None = None
    per_slot_winrate: dict[str, float] | None = None


def _run_single_simulation(
    simulation: Any,
    policy_data: Sequence[Any] | None,
    replay_dir: str | None,
    seed: int,
    device_override: str | None = None,
) -> SimulationRunResult:
    sim_cfg = SimulationRunConfig.model_validate(simulation)
    policy_specs = [PolicySpec.model_validate(spec) for spec in policy_data] if policy_data else []

    env_interface = PolicyEnvInterface.from_mg_cfg(sim_cfg.env)

    multi_agent_policies: list[MultiAgentPolicy]
    agent_map: Sequence[str] | None = None

    if sim_cfg.policy_slots:
        registry = SlotRegistry()
        slots_cfg = list(sim_cfg.policy_slots)
        slot_lookup: dict[str, int] = {}
        for slot_cfg in slots_cfg:
            if slot_cfg.id in slot_lookup:
                raise ValueError(f"Duplicate policy slot id '{slot_cfg.id}'")
            slot_lookup[slot_cfg.id] = len(slot_lookup)
        controller_device = torch.device(device_override or "cpu")

        num_agents = env_interface.num_agents
        if sim_cfg.agent_slot_map is not None:
            agent_map = list(sim_cfg.agent_slot_map)
        else:
            agent_map = [slots_cfg[0].id] * num_agents
        if len(agent_map) != num_agents:
            raise ValueError(f"agent_slot_map must match num_agents ({num_agents}); got {len(agent_map)}")
        slot_ids = []
        for idx, slot_id in enumerate(agent_map):
            if slot_id not in slot_lookup:
                raise ValueError(f"agent_slot_map[{idx}] references unknown slot id '{slot_id}'")
            slot_ids.append(slot_lookup[slot_id])
        agent_slot_tensor = torch.tensor(slot_ids, dtype=torch.long, device=controller_device)

        slot_policies = {
            idx: registry.get(
                slot_cfg,
                env_interface,
                device=controller_device,
            )
            for idx, slot_cfg in enumerate(slots_cfg)
        }
        controller = SlotControllerPolicy(
            slot_lookup=slot_lookup,
            slots=slots_cfg,
            slot_policies=slot_policies,
            policy_env_info=env_interface,
            controller_device=controller_device,
            agent_slot_map=agent_slot_tensor,
        )
        multi_agent_policies = [controller]
    else:
        if not policy_specs:
            raise ValueError("policy_specs required when no policy_slots provided")
        multi_agent_policies = [
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

    per_slot_returns = None
    per_slot_winrate = None
    if sim_cfg.policy_slots and rollout_result.episode_returns:
        effective_agent_map = agent_map or []
        if effective_agent_map:
            per_slot_returns = {}
            per_slot_winrate = {}
            returns_tensor = torch.tensor(rollout_result.episode_returns)
            wins_tensor = None
            if rollout_result.episode_wins is not None:
                wins_tensor = torch.tensor(rollout_result.episode_wins)
            for slot_id in set(effective_agent_map):
                idxs = [idx for idx, agent_slot in enumerate(effective_agent_map) if agent_slot == slot_id]
                if idxs:
                    per_slot_returns[slot_id] = float(returns_tensor[:, idxs].mean().item())
                    if wins_tensor is not None:
                        per_slot_winrate[slot_id] = float(wins_tensor[:, idxs].float().mean().item())

    return SimulationRunResult(
        run=sim_cfg,
        results=rollout_result,
        per_slot_returns=per_slot_returns,
        per_slot_winrate=per_slot_winrate,
    )


def run_simulations(
    *,
    policy_specs: Sequence[PolicySpec] | None,
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    max_workers: int | None = None,
    on_progress: Callable[[str], None] = lambda x: None,
    device_override: str | None = None,
) -> list[SimulationRunResult]:
    if not policy_specs and not any(sim.policy_slots for sim in simulations):
        raise ValueError("At least one policy spec or simulation-level policy slot is required")

    if not max_workers or max_workers <= 1 or len(simulations) <= 1:
        sequential_rollouts: list[SimulationRunResult] = []
        for i, simulation in enumerate(simulations):
            on_progress(f"Beginning rollout for simulation {i + 1} of {len(simulations)}")
            sequential_rollouts.append(
                _run_single_simulation(simulation, policy_specs, replay_dir, seed, device_override)
            )
            on_progress(f"Finished rollout for simulation {i + 1} of {len(simulations)}")

        return sequential_rollouts

    simulation_rollouts: list[SimulationRunResult] = [None] * len(simulations)  # type: ignore[assignment]

    simulation_payloads = [sim.model_dump(mode="json") for sim in simulations]
    policy_payloads = [spec.model_dump(mode="json") for spec in policy_specs] if policy_specs else []

    on_progress(f"Launching {len(simulations)} eval rollouts with up to {max_workers} workers")

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
