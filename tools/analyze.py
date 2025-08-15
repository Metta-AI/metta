#!/usr/bin/env -S uv run
"""Analysis tool for MettaGrid evaluation results."""

import logging

from pydantic import Field

from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff, WandbConfigOn
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.rl.system_config import SystemConfig
from metta.util.metta_script import pydantic_metta_script

logger = logging.getLogger("analyze")


class AnalysisToolConfig(Config):
    analysis: AnalysisConfig
    policy_uri: str
    system: SystemConfig = Field(default_factory=SystemConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfigOff)
    data_dir: str = Field(default="./train_dir")


def main(cfg: AnalysisToolConfig) -> None:
    policy_store = PolicyStore(
        device=cfg.system.device,
        data_dir=cfg.data_dir,
        wandb_entity=cfg.wandb.entity if isinstance(cfg.wandb, WandbConfigOn) else None,
        wandb_project=cfg.wandb.project if isinstance(cfg.wandb, WandbConfigOn) else None,
    )
    policy_pr = policy_store.policy_record(
        cfg.policy_uri, cfg.analysis.policy_selector.type, metric=cfg.analysis.policy_selector.metric
    )
    analyze(policy_pr, cfg.analysis)


pydantic_metta_script(main)
