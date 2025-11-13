import logging
from typing import Sequence

from pydantic import Field

from metta.common.tool import Tool
from metta.sim.runner import FullSimulationConfig, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.loader import resolve_policy_class_path

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    simulations: Sequence[FullSimulationConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    def invoke(self, args: dict[str, str]) -> int | None:
        # This is closer to what is being done elsewhere, but we may want to shove the resolving of the policy
        # class into initialize_and_load_policy itself
        for simulation in self.simulations:
            for policy_spec in simulation.policy_specs:
                policy_spec.class_path = resolve_policy_class_path(policy_spec.class_path)

        simulation_rollouts = run_simulations(
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        logger.info(f"Simulation rollouts: {simulation_rollouts}")
        return 0
