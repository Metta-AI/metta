#!/usr/bin/env -S uv run
"""Generate dashboard data."""

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.eval.dashboard_data import DashboardConfig, write_dashboard_data
from metta.mettagrid.util.file import http_url

DASHBOARD_URL = "https://metta-ai.github.io/metta/observatory/"


@hydra.main(version_base=None, config_path="../configs", config_name="dashboard_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("dashboard")

    logger.info(f"Dashboard job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("Generating dashboard")

    config = DashboardConfig(cfg.dashboard)

    write_dashboard_data(config)

    if config.output_path.startswith("s3://"):
        logger.info(
            "Wrote dashboard data to S3. View dashboard at " + DASHBOARD_URL + "?data=" + http_url(config.output_path)
        )
    else:
        logger.info(
            f"Wrote dashboard data to {config.output_path}. Upload the data to " + DASHBOARD_URL + " to visualize"
        )


if __name__ == "__main__":
    main()
