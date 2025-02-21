import logging
import os

import hydra
from agent.policy_store import PolicyStore
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rl.wandb.wandb_context import WandbContext
import torch.multiprocessing
import torch.distributed

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

    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if cfg.trainer.num_gpus > 1:
        torch.multiprocessing.spawn(train_ddp,
            args=(cfg,),
            nprocs=cfg.trainer.num_gpus,
            join=True,
        )
    else:
        train(cfg)


def train_ddp(device_id, cfg):
    print(f"Training on {device_id}/{cfg.trainer.num_gpus} GPUs")
    cfg.device = f'{cfg.device}:{device_id}'
    torch.distributed.init_process_group(backend='nccl', rank=device_id, world_size=cfg.trainer.num_gpus)
    train(cfg)
    torch.distributed.destroy_process_group()

def train(cfg):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
        trainer.train()
        trainer.close()

if __name__ == "__main__":
    main()
