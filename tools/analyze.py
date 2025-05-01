"""Analysis tool for MettaGrid evaluation results."""

import hydra
from omegaconf import DictConfig

from metta.eval.report import dump_stats, generate_report
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("analyze")

    view_type = "latest"
    logger.info(f"Generating {view_type} report")
    dump_stats(cfg)
    generate_report(cfg)


if __name__ == "__main__":
    main()
