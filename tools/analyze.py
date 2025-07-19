#!/usr/bin/env -S uv run
"""Analysis tool for MettaGrid evaluation results."""

import logging

from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.util.metta_script import metta_script

logger = logging.getLogger("analyze")


def main(cfg: DictConfig) -> None:
    logger.info(f"Analyze job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    config = AnalysisConfig(cfg.analysis)

    policy_store = PolicyStore(cfg, None)
    policy_pr = policy_store.policy_record(
        config.policy_uri, config.policy_selector.type, metric=config.policy_selector.metric
    )
    analyze(policy_pr, config)


metta_script(main, "analyze_job")
