import logging
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.rl.binding_config import PolicyBindingConfig
from metta.rl.binding_controller import BindingControllerPolicy
from metta.rl.policy_registry import PolicyRegistry
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
    policy_bindings: Sequence[PolicyBindingConfig] | None = None
    agent_binding_map: Sequence[str] | None = None

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
    policy_specs: Sequence[PolicySpec] | None,
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str | None,
    seed: int,
    on_progress: Callable[[str], None] = lambda x: None,
) -> list[SimulationRunResult]:
    if not policy_specs and not any(sim.policy_bindings for sim in simulations):
        raise ValueError("At least one policy spec or simulation-level policy binding is required")

    simulation_rollouts: list[SimulationRunResult] = []

    for i, simulation in enumerate(simulations):
        proportions = simulation.proportions

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = []

        # Prefer simulation-specific bindings if provided; otherwise use policy_specs
        if simulation.policy_bindings:
            registry = PolicyRegistry()
            bindings_cfg = simulation.policy_bindings
            binding_lookup = {b.id: idx for idx, b in enumerate(bindings_cfg)}
            binding_policies = {
                idx: registry.get(
                    b,
                    env_interface,
                    device="cpu",  # sim runs default to CPU; extend later if needed
                )
                for idx, b in enumerate(bindings_cfg)
            }
            controller = BindingControllerPolicy(
                binding_lookup=binding_lookup,
                bindings=bindings_cfg,
                binding_policies=binding_policies,
                policy_env_info=env_interface,
                device="cpu",
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

        simulation_rollouts.append(
            SimulationRunResult(
                run=simulation,
                results=rollout_result,
            )
        )

    return simulation_rollouts
