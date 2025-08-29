"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.rl.checkpoint_manager import get_checkpoint_from_dir
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class AnalysisTool(Tool):
    wandb: WandbConfig = auto_wandb_config()

    analysis: AnalysisConfig
    policy_uri: str
    data_dir: str = Field(default="./train_dir")

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        # Pass URI directly to analyze function
        if self.policy_uri.startswith("file://"):
            # Verify the path exists
            checkpoint_dir = self.policy_uri[7:]  # Remove "file://" prefix
            result = get_checkpoint_from_dir(checkpoint_dir)
            if result is None:
                logger.error(f"No checkpoints found in directory: {checkpoint_dir}")
                return 1
            # Use the URI from the result
            policy_uri = result[0]
        else:
            # Use the URI as-is for wandb:// or other schemes
            policy_uri = self.policy_uri

        analyze(policy_uri, self.analysis)
        return 0
