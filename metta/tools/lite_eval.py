import logging
from typing import Sequence

from pydantic import Field

from metta.common.tool import Tool
from metta.sim.runner import run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.loader import resolve_policy_class_path
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    simulations: Sequence[SimulationConfig] = Field(description="Simulations to evaluate")
    policies: Sequence[PolicySpec] = Field(description="Policies to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_rollouts = run_simulations(
            simulations=self.simulations,
            policies=[
                PolicySpec(
                    class_path=resolve_policy_class_path(s.class_path),
                    data_path=s.data_path,
                    proportion=s.proportion,
                )
                for s in self.policies
            ],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        logger.info(f"Simulation rollouts: {simulation_rollouts}")
        return 0
