import argparse
import logging
import os
import sys
import hydra
from typing import List, Optional

import yaml
from agent.policy_store import PolicyStore
from util.runtime_configuration import setup_metta_environment, setup_omega_conf
from omegaconf import OmegaConf, open_dict, DictConfig
from rich.logging import RichHandler
from rl.wandb.wandb_context import WandbContext
import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(processName)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("train")

def train(cfg, wandb_run):
    # pretty print the cfg as yaml
    logger.info("Training with config:")
    logger.info(OmegaConf.to_yaml(cfg))
    policy_store = PolicyStore(cfg, wandb_run)
    trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
    trainer.train()
    trainer.close()

@record
@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: OmegaConf) -> int:
    wandb_run_id = None
    if cfg.dist_cfg_path is not None:
        dist_cfg = OmegaConf.load(cfg.dist_cfg_path)
        cfg.run = dist_cfg.run
        wandb_run_id = dist_cfg.wandb_run_id
    setup_metta_environment(cfg)

    if "LOCAL_RANK" in os.environ and cfg.device == "cuda":
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f'{cfg.device}:{local_rank}'
        dist.init_process_group(backend="nccl")

    if os.environ.get("RANK", "0") == "0":
        overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
        if os.path.exists(overrides_path):
            logger.info(f"Loading train config overrides from {overrides_path}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(overrides_path))

        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        with WandbContext(cfg, run_id=wandb_run_id) as wandb_run:
            train(cfg, wandb_run)
    else:
        train(cfg, None)

if __name__ == "__main__":
    sys.exit(main())
