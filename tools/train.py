import logging
import os
import datetime
import hydra
from agent.policy_store import PolicyStore
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rl.wandb.wandb_context import WandbContext
import torch.multiprocessing
import torch.distributed

# Configure rich colored logging
FORMAT = "%(asctime)s %(processName)s %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)

logger = logging.getLogger("train")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    with WandbContext(cfg) as wandb_run:
        if cfg.trainer.dist.num_gpus > 1:
            # Enable logging in subprocesses
            torch.multiprocessing.set_start_method('spawn', force=True)
            torch.multiprocessing.log_to_stderr = True

            torch.multiprocessing.spawn(train_ddp,
                args=(wandb_run, cfg),
                nprocs=cfg.trainer.dist.num_gpus,
                join=True,
            )
        else:
            train(wandb_run, cfg)


def train_ddp(device_id, wandb_run, cfg):
    setup_metta_environment(cfg)
    print(f"Training on {device_id}/{cfg.trainer.dist.num_gpus} GPUs")
    cfg.device = f'{cfg.device}:{device_id}'
    torch.distributed.init_process_group(
        backend='nccl',
        rank=device_id,
        world_size=cfg.trainer.dist.num_gpus,
        # timeout=datetime.timedelta(seconds=cfg.trainer.dist.nccl.timeout),
    )
    logger.info(f"train_ddp() on {device_id}")
    train(wandb_run, cfg)
    torch.distributed.destroy_process_group()

def train(wandb_run, cfg):
    setup_metta_environment(cfg)
    policy_store = PolicyStore(cfg, wandb_run)

    logger.info(f"making trainer on {cfg.device}")
    print(f"making trainer on {cfg.device}")
    trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
    logger.info(f"train.start() on {trainer.device}")
    print(f"train.start() on {trainer.device}")
    trainer.train()
    trainer.close()

if __name__ == "__main__":
    main()
