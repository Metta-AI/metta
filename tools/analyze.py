#!/usr/bin/env -S uv run
"""Analysis tool for MettaGrid evaluation results."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from metta.util.metta_script import setup_mettagrid_environment

logger = logging.getLogger("analyze")


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    setup_mettagrid_logger()

    logger.info(f"Analyze job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    config = AnalysisConfig(cfg.analysis)

    policy_store = PolicyStore(cfg, None)
    policy_pr = policy_store.policy_record(
        config.policy_uri, config.policy_selector.type, metric=config.policy_selector.metric
    )
    analyze(policy_pr, config)


if __name__ == "__main__":
    main()
