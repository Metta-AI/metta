import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import Field

from metta.common.tool import Tool
from metta.sim.replay_log_writer import ReplayLogWriter
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode_rollout import MultiEpisodeRolloutResult, multi_episode_rollout

logger = logging.getLogger(__name__)


@dataclass
class SimulationRollout:
    simulation: SimulationConfig
    rollout: MultiEpisodeRolloutResult
    replay_urls: dict[str, str]


class NewEvalTool(Tool):
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policies: Sequence[PolicySpec] = Field(description="Policies to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)
    enable_replays: bool = True

    def _create_replay_writer(self, simulation: SimulationConfig) -> ReplayLogWriter | None:
        if not self.enable_replays:
            return None
        replay_root = Path(self.replay_dir).expanduser()
        unique_dir = replay_root / simulation.suite / simulation.name / uuid.uuid4().hex[:12]
        return ReplayLogWriter(str(unique_dir))

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_rollouts: list[SimulationRollout] = []
        policy_proportions = [spec.proportion for spec in self.policies]

        for simulation in self.simulations:
            env_interface = PolicyEnvInterface.from_mg_cfg(simulation.env)
            multi_agent_policies: list[MultiAgentPolicy] = [
                initialize_or_load_policy(env_interface, spec) for spec in self.policies
            ]

            replay_writer = self._create_replay_writer(simulation)
            rollout_result = multi_episode_rollout(
                env_cfg=simulation.env,
                policies=multi_agent_policies,
                episodes=simulation.num_episodes,
                seed=self.system.seed,
                proportions=policy_proportions,
                max_time_s=simulation.max_time_s,
                max_action_time_ms=simulation.max_action_time_ms,
                event_handlers=[replay_writer] if replay_writer else None,
            )

            replay_urls = replay_writer.get_written_replay_urls() if replay_writer else {}
            if replay_urls:
                logger.info("Captured %d replay(s) for %s", len(replay_urls), simulation.full_name)

            simulation_rollouts.append(
                SimulationRollout(
                    simulation=simulation,
                    rollout=rollout_result,
                    replay_urls=replay_urls,
                )
            )

        self.results = simulation_rollouts

        # TODO: Log to wandb
        return 0
