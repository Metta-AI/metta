import logging
from typing import Sequence

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    policy_specs: Sequence[PolicySpec] = Field(description="Policy specs to evaluate")
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    @model_validator(mode="after")
    def validate(self) -> "LiteEvalTool":
        for simulation in self.simulations:
            if simulation.proportions is not None and len(simulation.proportions) != len(self.policy_specs):
                raise ValueError("Number of proportions must match number of policies.")
        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_results = run_simulations(
            policy_specs=self.policy_specs,
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        logger.info(f"Simulation results: {simulation_results}")

        summaries = build_multi_episode_rollout_summaries(
            rollout_results=[result.results for result in simulation_results],
            policy_specs=list(self.policy_specs),
        )
        logger.info(f"Summaries: {summaries}")
        return 0
