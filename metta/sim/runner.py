import logging
from typing import Callable, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field

from metta.rl.slot_config import PolicySlotConfig
from metta.rl.slot_controller import SlotControllerPolicy
from metta.rl.slot_registry import SlotRegistry
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


def run_simulations(
    *,
    policy_specs: Sequence[PolicySpec] | None,
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    on_progress: Callable[[str], None] = lambda x: None,
) -> list[SimulationRunResult]:
    if not policy_specs and not any(sim.policy_slots for sim in simulations):
        raise ValueError("At least one policy spec or simulation-level policy slot is required")

    simulation_rollouts: list[SimulationRunResult] = []

    for i, simulation in enumerate(simulations):
        proportions = simulation.proportions

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = []

        # Prefer simulation-specific bindings if provided; otherwise use policy_specs
        if simulation.policy_slots:
            registry = SlotRegistry()
            slots_cfg = simulation.policy_slots
            slot_lookup = {b.id: idx for idx, b in enumerate(slots_cfg)}
            # Build agent slot map tensor
            num_agents = env_interface.num_agents
            agent_map = simulation.agent_slot_map or [slots_cfg[0].id for _ in range(num_agents)]
            if len(agent_map) != num_agents:
                raise ValueError(f"agent_slot_map must match num_agents ({num_agents}); got {len(agent_map)}")
            slot_ids = [slot_lookup[a] for a in agent_map]
            agent_slot_tensor = torch.tensor(slot_ids, dtype=torch.long)

            slot_policies = {
                idx: registry.get(
                    b,
                    env_interface,
                    device="cpu",  # sim runs default to CPU; extend later if needed
                )
                for idx, b in enumerate(slots_cfg)
            }
            controller = SlotControllerPolicy(
                slot_lookup=slot_lookup,
                slots=slots_cfg,
                slot_policies=slot_policies,
                policy_env_info=env_interface,
                device="cpu",
                agent_slot_map=agent_slot_tensor,
            )
            multi_agent_policies.append(controller)
        else:
            assert policy_specs is not None
            multi_agent_policies = [initialize_or_load_policy(env_interface, spec) for spec in policy_specs]

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

        per_slot_returns = None
        if simulation.policy_slots and rollout_result.episode_returns:
            # Compute average return per slot by agent index mapping
            agent_map = simulation.agent_slot_map or []
            if agent_map:
                per_slot_returns = {}
                # episode_returns shape: [num_episodes, num_agents]
                returns_tensor = torch.tensor(rollout_result.episode_returns)
                for slot_id in set(agent_map):
                    idxs = [j for j, b in enumerate(agent_map) if b == slot_id]
                    if idxs:
                        per_slot_returns[slot_id] = float(returns_tensor[:, idxs].mean().item())

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
                per_slot_returns=per_slot_returns,
            )
        )

    return simulation_rollouts
