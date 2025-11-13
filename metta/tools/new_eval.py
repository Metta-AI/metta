import logging
from dataclasses import dataclass
from typing import Sequence

from pydantic import Field

from metta.common.tool import Tool
from metta.sim.runner import run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode_rollout import MultiEpisodeRolloutResult

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

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_rollouts = run_simulations(
            simulations=self.simulations,
            policies=self.policies,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=self.enable_replays,
        )
        self.results = simulation_rollouts
        return 0
