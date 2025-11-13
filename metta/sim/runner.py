import uuid
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from metta.sim.replay_log_writer import ReplayLogWriter
from metta.sim.simulation_config import SimulationConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode_rollout import MultiEpisodeRolloutResult, multi_episode_rollout


class SimulationRollout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation: SimulationConfig
    rollout: MultiEpisodeRolloutResult
    replay_urls: dict[str, str]


def run_simulations(
    simulations: Sequence[SimulationConfig],
    policies: Sequence[PolicySpec],
    replay_dir: str,
    seed: int,
    enable_replays: bool = True,
    proportions: Sequence[float] | None = None,
) -> list[SimulationRollout]:
    simulation_rollouts: list[SimulationRollout] = []

    for simulation in simulations:
        replay_writer: ReplayLogWriter | None = None
        if enable_replays:
            replay_root = Path(replay_dir).expanduser()
            unique_dir = replay_root / simulation.suite / simulation.name / uuid.uuid4().hex[:12]
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
            max_time_s=simulation.max_time_s,
            max_action_time_ms=simulation.max_action_time_ms,
            event_handlers=[replay_writer] if replay_writer else None,
        )

        replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}

        simulation_rollouts.append(
            SimulationRollout(
                simulation=simulation,
                rollout=rollout_result,
                replay_urls=replay_urls,
            )
        )

    return simulation_rollouts
