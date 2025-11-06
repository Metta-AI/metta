import logging

import pydantic

import metta.common.tool
import metta.common.wandb.context
import metta.eval.analysis
import metta.eval.analysis_config
import metta.tools.utils.auto_config

logger = logging.getLogger(__name__)


class AnalysisTool(metta.common.tool.Tool):
    wandb: metta.common.wandb.context.WandbConfig = metta.tools.utils.auto_config.auto_wandb_config()

    analysis: metta.eval.analysis_config.AnalysisConfig
    policy_uri: str
    data_dir: str = pydantic.Field(default="./train_dir")

    def invoke(self, args: dict[str, str]) -> int | None:
        metta.eval.analysis.analyze(self.policy_uri, self.analysis)
        return 0
