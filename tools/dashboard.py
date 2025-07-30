#!/usr/bin/env -S uv run
"""Generate dashboard data."""

import logging

from omegaconf import DictConfig, OmegaConf

from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.eval.dashboard_data import DashboardConfig, write_dashboard_data
from metta.mettagrid.util.file import http_url
from metta.util.metta_script import metta_script

DASHBOARD_URL = f"{METTASCOPE_REPLAY_URL}/observatory/"


logger = logging.getLogger("dashboard")


def main(cfg: DictConfig) -> None:
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


metta_script(main, "dashboard_job")
