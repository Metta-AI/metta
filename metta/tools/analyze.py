"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.rl.checkpoint_interface import Checkpoint, get_checkpoint_from_dir
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class AnalysisTool(Tool):
    wandb: WandbConfig = auto_wandb_config()

    analysis: AnalysisConfig
    policy_uri: str
    data_dir: str = Field(default="./train_dir")

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        # Nuclear simplification - direct checkpoint loading from URI
        if self.policy_uri.startswith("file://"):
            # Handle file:// URIs by extracting directory path
            checkpoint_dir = self.policy_uri[7:]  # Remove "file://" prefix
            checkpoint = get_checkpoint_from_dir(checkpoint_dir)
            if checkpoint is None:
                logger.error(f"No checkpoints found in directory: {checkpoint_dir}")
                return 1
        else:
            # For other URIs, create minimal checkpoint object
            logger.warning(f"Non-file URI {self.policy_uri} - creating minimal checkpoint")
            checkpoint = Checkpoint(run_name="unknown", uri=self.policy_uri, metadata={"epoch": 0})

        analyze(checkpoint, self.analysis)
