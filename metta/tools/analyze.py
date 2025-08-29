import logging

from pydantic import Field

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
        if self.policy_uri.startswith("file://"):
            # CheckpointManager.load_from_uri handles directories automatically
            policy_uri = self.policy_uri
        else:
            policy_uri = self.policy_uri

        analyze(policy_uri, self.analysis)
        return 0
