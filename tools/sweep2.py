import logging
import os
import signal  # Aggressively exit on ctrl+c

import hydra
import torch.distributed as dist
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
from wandb_carbs import create_sweep

from agent.policy_store import PolicyStore
from rl.carbs.metta_carbs import carbs_params_from_cfg
from rl.carbs.rollout import CarbsSweepRollout
from rl.wandb.sweep import sweep_id_from_name
from rl.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


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
    policy_store = PolicyStore(cfg, wandb_run)
    trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
    trainer.train()
    trainer.close()

def init_sweep(cfg):
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.run)
    if not sweep_id:
        logger.debug(f"Sweep {cfg.run} not found, creating new sweep")
        os.makedirs(os.path.join(cfg.run_dir, "runs"), exist_ok=True)

        sweep_id = create_sweep(
            cfg.run,
            cfg.wandb.entity,
            cfg.wandb.project,
            carbs_params_from_cfg(cfg)[0]
        )

        logger.debug(f"WanDb Sweep created with ID: {sweep_id}")
    return sweep_id
@record
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    if os.environ.get("RANK", "0") == "0":
        sweep_id =  init_sweep(cfg)
        with WandbContext(cfg) as wandb_run:
            rollout = CarbsSweepRollout(cfg, wandb_run, sweep_id)
    else:
        rollout = CarbsSweepRollout(cfg, None, None)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f'{cfg.device}:{local_rank}'
        dist.init_process_group(backend="nccl")

    rollout.run()


if __name__ == "__main__":
    main()
