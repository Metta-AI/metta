"""
Adapter to run the dashboard from a Hydra configuration.
This allows integrating the dashboard with existing Hydra-based workflows.
"""

import logging
import hydra
from omegaconf import DictConfig
from sim.report.app import App

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def dashboard_tool(cfg: DictConfig) -> None:
    logger.info("Starting dashboard with Hydra configuration")
    
    # Extract dashboard-specific config if needed
    if 'dashboard' in cfg:
        dashboard_cfg = cfg.dashboard
    else:
        dashboard_cfg = cfg
    
    app = App(dashboard_cfg)
    app.run()

if __name__ == "__main__":
    dashboard_tool()