import logging

from pydantic import Field

from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class AnalysisTool(Tool):
    wandb: WandbConfig = auto_wandb_config()

    analysis: AnalysisConfig
    policy_uri: str
    data_dir: str = Field(default="./train_dir")

    def invoke(self, args: dict[str, str]) -> int | None:
        analyze(self.policy_uri, self.analysis)
        return 0
