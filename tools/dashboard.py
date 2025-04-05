"""
Adapter to run the dashboard from a Hydra configuration.
This allows integrating the dashboard with existing Hydra-based workflows.
"""

import logging
import hydra
from omegaconf import DictConfig
from sim.eval import DashboardApp, DashboardConfig

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")
def serve_dashboard(cfg: DictConfig) -> None:
    logger.info("Starting dashboard with Hydra configuration")
    
    if 'dashboard' in cfg:
        dashboard_cfg = DashboardConfig.from_dict_config(cfg.dashboard)
    else:
        dashboard_cfg = DashboardConfig()

    app = DashboardApp(dashboard_cfg, cfg)
    app.run()

if __name__ == "__main__":
    serve_dashboard()