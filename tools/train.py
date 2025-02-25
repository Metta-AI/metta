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
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
    force=True
)

logger = logging.getLogger("train")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    logger.info("Setting up distributed environment")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    logger.info(f"Initializing multi-GPU training with {cfg.trainer.dist.num_gpus} GPUs")
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.log_to_stderr = True

    try:
        logger.info("Spawning distributed processes")
        torch.multiprocessing.spawn(train_ddp,
            args=(cfg,),
            nprocs=cfg.trainer.dist.num_gpus,
            join=True,
        )
        logger.info("All distributed processes completed")
    except Exception as e:
        logger.error(f"Error in multiprocessing: {e}")
        raise


def train_ddp(device_id, cfg):
    # Reconfigure logging for each process
    logging.basicConfig(
        level="INFO",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
        force=True
    )
    logger = logging.getLogger("train")

    logger.info(f"Starting train_ddp on device {device_id}")
    cfg.device = f'{cfg.device}:{device_id}'

    logger.info(f"Initializing process group for device {device_id}")
    torch.distributed.init_process_group(
        backend='nccl',
        rank=device_id,
        world_size=cfg.trainer.dist.num_gpus,
        timeout=datetime.timedelta(seconds=cfg.trainer.dist.nccl.timeout),
    )
    logger.info(f"Process group initialized for device {device_id}")

    try:
        logger.info(f"Starting training on device {device_id}")
        train(cfg)
        logger.info(f"Training completed on device {device_id}")
    except Exception as e:
        logger.error(f"Error in train_ddp on device {device_id}: {e}")
        raise
    finally:
        logger.info(f"Cleaning up process group for device {device_id}")
        torch.distributed.destroy_process_group()


def train(cfg):
    logger.info(f"Starting train() on device {cfg.device}")
    setup_metta_environment(cfg)

    raise Exception("Stop here")
    logger.info(f"Instantiating trainer on {cfg.device}")
    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
        logger.info(f"Starting trainer.train() on {trainer.device}")
        trainer.train()
        trainer.close()
        logger.info(f"Training completed on {trainer.device}")

if __name__ == "__main__":
    main()
