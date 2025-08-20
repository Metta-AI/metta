"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

from metta.agent.policy_store import PolicyStore
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

    def invoke(self) -> None:
        policy_store = PolicyStore(
            device=self.system.device,
            data_dir=self.data_dir,
            wandb_entity=self.wandb.entity if self.wandb.enabled else None,
            wandb_project=self.wandb.project if self.wandb.enabled else None,
        )
        policy_pr = policy_store.policy_record(
            self.policy_uri, self.analysis.policy_selector.type, metric=self.analysis.policy_selector.metric
        )
        analyze(policy_pr, self.analysis)
