import logging
from typing import Sequence

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.sim.handle_results import render_eval_summary
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)


class MultiPolicyEvalTool(Tool):
    policy_specs: Sequence[PolicySpec] = Field(description="Policy specs to evaluate")
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)
    verbose: bool = Field(default=True, description="Whether to log verbose output")
    log_to_wandb: bool = Field(default=False, description="Whether to log to wandb")

    @model_validator(mode="after")
    def validate(self) -> "MultiPolicyEvalTool":
        for simulation in self.simulations:
            if simulation.proportions is not None and len(simulation.proportions) != len(self.policy_specs):
                raise ValueError("Number of proportions must match number of policies.")
        if not self.policy_specs:
            raise ValueError("No policy specs provided")
        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_results = run_simulations(
            policy_specs=self.policy_specs,
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
        )
        render_eval_summary(
            simulation_results, policy_names=[spec.name for spec in self.policy_specs], verbose=self.verbose
        )
        return 0
