import logging

import hydra
from omegaconf import DictConfig
from rich.logging import RichHandler

from eval import simulate_policy
from rl.wandb.wandb_context import WandbContext

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(processName)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("eval")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    if not cfg.wandb.enabled:
        logger.info(f"run {cfg.run}: agent.PolicyStore needs more work to run locally")

    else:
        with WandbContext(cfg) as wandb_run:
            simulate_policy(cfg, wandb_run)


if __name__ == "__main__":
    main()
