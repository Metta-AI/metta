"""Generate a heatmap."""

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.eval.dashboard.dashboard_config import DashboardConfig
from metta.eval.dashboard.page import generate_dashboard
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment


@hydra.main(version_base=None, config_path="../configs", config_name="dashboard_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("dashboard")

    logger.info(f"Dashboard job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("Generating dashboard")

    config = DashboardConfig(cfg.dashboard)

    generate_dashboard(config)


if __name__ == "__main__":
    main()
