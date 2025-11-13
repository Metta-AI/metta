import uuid
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from metta.sim.replay_log_writer import ReplayLogWriter
from mettagrid import MettaGridConfig
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode_rollout import MultiEpisodeRolloutResult, multi_episode_rollout


class EnvRunConfig(Config):
    env: MettaGridConfig  # noqa: F821
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)

    max_action_time_ms: int | None = Field(
        default=10000, description="Maximum time (in ms) a policy is given to take an action"
    )
    episode_tags: Optional[list[str]] = Field(default=None, description="Tags to add to each episode")


class FullSimulationConfig(BaseModel):
    env_run: EnvRunConfig
    policy_specs: Sequence[PolicySpec]
    proportions: Sequence[float] | None = None


class SimulationRollout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_run: EnvRunConfig
    rollout: MultiEpisodeRolloutResult
    replay_urls: dict[str, str]


def run_simulations(
    simulations: Sequence[FullSimulationConfig],
    replay_dir: str,
    seed: int,
    enable_replays: bool = True,
) -> list[SimulationRollout]:
    simulation_rollouts: list[SimulationRollout] = []

    for full_simulation in simulations:
        simulation = full_simulation.env_run
        policies = full_simulation.policy_specs
        proportions = full_simulation.proportions
        replay_writer: ReplayLogWriter | None = None
        if enable_replays:
            replay_root = Path(replay_dir).expanduser()
            unique_dir = replay_root / uuid.uuid4().hex[:12]
            replay_writer = ReplayLogWriter(str(unique_dir))

        env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
        multi_agent_policies: list[MultiAgentPolicy] = [
            initialize_or_load_policy(env_interface, spec) for spec in policies
        ]

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

        replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}

        simulation_rollouts.append(
            SimulationRollout(
                env_run=simulation,
                rollout=rollout_result,
                replay_urls=replay_urls,
            )
        )

    return simulation_rollouts
