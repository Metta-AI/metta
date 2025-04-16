import logging
import os
import signal  # Aggressively exit on ctrl+c

import hydra
from rich.logging import RichHandler

from metta.agent.policy_store import PolicyStore
from metta.rl.pufferlib.play import play
from metta.util.config import config_from_path, setup_metta_environment
from metta.util.wandb.wandb_context import WandbContext

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(processName)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("eval")

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    if not cfg.wandb.enabled:
        logger.info(f"run {cfg.run}: agent.PolicyStore needs more work to run locally")

    else:
        setup_metta_environment(cfg)
        cfg.eval.env = config_from_path(cfg.eval.env, cfg.eval.env_overrides)

        with WandbContext(cfg) as wandb_run:
            policy_store = PolicyStore(cfg, wandb_run)
            play(cfg, policy_store)


if __name__ == "__main__":
    main()
