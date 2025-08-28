"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

# TODO: Update to use SimpleCheckpointManager instead of PolicyStore
# from metta.rl.simple_checkpoint_manager import SimpleCheckpointManager
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
        # TODO: Update this to use SimpleCheckpointManager
        # For now, keeping PolicyStore to avoid breaking the analysis tools
        from metta.agent.policy_store import PolicyStore

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
