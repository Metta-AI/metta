import os
import signal  # Aggressively exit on ctrl+c
import logging

import hydra
from omegaconf import OmegaConf
from rich import traceback
from rich.console import Console
from rich.logging import RichHandler

from rl.pufferlib.trainer import PufferTrainer
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything


signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("train")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)

    logger.info(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    with WandbContext(cfg) as wandb_run:
        # trainer = PufferTrainer(cfg, wandb_run)
        trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run)
        trainer.train()
        trainer.close()

if __name__ == "__main__":
    main()
