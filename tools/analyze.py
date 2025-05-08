"""Analysis tool for MettaGrid evaluation results."""

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.eval.analysis_config import AnalyzerConfig
from metta.eval.dashboard.page import generate_report
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("analyze")

    logger.info(f"Analyze job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    analyzer = AnalyzerConfig(cfg.analyzer)

    generate_report(analyzer)


if __name__ == "__main__":
    main()
