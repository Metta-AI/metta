"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

from metta.agent.policy_finder import PolicyFinder
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class AnalysisTool(Tool):
    wandb: WandbConfig = auto_wandb_config()

    analysis: AnalysisConfig
    policy_uri: str
    data_dir: str = Field(default="./train_dir")

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        policy_finder = PolicyFinder.create(
            wandb_entity=self.wandb.entity if self.wandb.enabled else None,
            wandb_project=self.wandb.project if self.wandb.enabled else None,
        )
        policy_pr = policy_finder.policy_records(
            self.policy_uri, self.analysis.policy_selector.type, metric=self.analysis.policy_selector.metric
        )[0]
        analyze(policy_pr, self.analysis)
