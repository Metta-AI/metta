import logging
from typing import Sequence

from pydantic import Field

from metta.common.tool import Tool
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_rollouts = run_simulations(
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        logger.info(f"Simulation rollouts: {simulation_rollouts}")
        return 0
