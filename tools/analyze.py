import logging

import hydra
from omegaconf import DictConfig

from eval.report import generate_report
from util.runtime_configuration import setup_mettagrid_environment

@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    logger = logging.getLogger(__name__)
    view_type = "latest"
    logger.info(f"Generating {view_type} report")
    generate_report(cfg)


if __name__ == "__main__":
    main()
