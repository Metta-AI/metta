import json
import logging
from datetime import datetime
from functools import partial
from typing import Sequence

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.common.wandb.context import WandbContext
from metta.eval.eval_request_config import EvalResults
from metta.rl import stats as rl_stats
from metta.sim.runner import SimulationRunConfig, build_eval_results, run_simulations
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri, auto_wandb_config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)


class LiteEvalTool(Tool):
    policy_specs: Sequence[PolicySpec] = Field(description="Policy specs to evaluate")
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)

    log_to_wandb: bool = Field(default=False, description="Whether to log to wandb")

    @model_validator(mode="after")
    def validate(self) -> "LiteEvalTool":
        for simulation in self.simulations:
            if simulation.proportions is not None and len(simulation.proportions) != len(self.policy_specs):
                raise ValueError("Number of proportions must match number of policies.")
        if not self.policy_specs:
            raise ValueError("No policy specs provided")
        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        simulation_results = run_simulations(
            policy_initializers=[partial(initialize_or_load_policy, policy_spec=spec) for spec in self.policy_specs],
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        eval_results = build_eval_results(simulation_results, 0)
        logger.info(
            "EvalResults schema: %s",
            json.dumps(
                {
                    "metrics": {
                        "reward_avg": eval_results.scores.avg_simulation_score,
                        "reward_avg_category_normalized": eval_results.scores.avg_category_score,
                        "detailed": eval_results.scores.to_wandb_metrics_format(),
                    },
                    "replay_url": eval_results.replay_urls,
                },
                indent=2,
            ),
        )
        if self.log_to_wandb:
            self._log_to_wandb(eval_results)
        return 0

    def _log_to_wandb(self, eval_results: EvalResults) -> None:
        wandb = auto_wandb_config(f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with WandbContext(wandb, self) as wandb_run:
            if not wandb_run:
                logger.info("Failed to initialize wandb run, skipping wandb logging")
                return
            logger.info(f"Initialized wandb run: {wandb_run.id}")

            policy_spec = self.policy_specs[0]
            target_policy_uri = f"file://{policy_spec.class_path}:{policy_spec.data_path}"

            rl_stats.process_policy_evaluator_stats(
                target_policy_uri, eval_results, wandb_run, epoch=0, agent_step=0, should_finish_run=True
            )
