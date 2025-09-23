import logging

from pydantic import Field

from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from pydantic import Field
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class AnalysisTool(Tool):
    wandb: WandbConfig = auto_wandb_config()

    analysis: AnalysisConfig
    policy_uri: str | None = None

    data_dir: str = Field(default="./train_dir")

    def invoke(self, args: dict[str, str]) -> int | None:
        # Use policy_uri from tool if provided, otherwise use from analysis config
        policy_uri = self.policy_uri or self.analysis.policy_uri
        analyze(policy_uri, self.analysis)
        return 0
