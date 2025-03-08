import logging
import os
import signal
import time  # Aggressively exit on ctrl+c

import hydra
import torch
import torch.distributed as dist
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
import wandb
from wandb_carbs import create_sweep

from agent.policy_store import PolicyStore
from rl.carbs.metta_carbs import carbs_params_from_cfg
from rl.carbs.rollout import MasterSweepRollout, WorkerSweepRollout
from rl.wandb.sweep import sweep_id_from_name
from rl.wandb.wandb_context import WandbContext
from rl.wandb.sweep import generate_run_id_for_sweep

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

global _cfg
def init_run():
    global _cfg
    with WandbContext(_cfg) as wandb_run:
        wandb_run.name = _cfg.run

def init_sweep(cfg):
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.sweep.name)
    if not sweep_id:
        logger.debug(f"Sweep {cfg.sweep.name} not found, creating new sweep")
        os.makedirs(os.path.join(cfg.run_dir, "runs"), exist_ok=True)

        sweep_id = create_sweep(
            cfg.sweep.name,
            cfg.wandb.entity,
            cfg.wandb.project,
            carbs_params_from_cfg(cfg)[0]
        )

        logger.debug(f"WanDb Sweep created with ID: {sweep_id}")

    cfg.sweep.id = sweep_id
    cfg.wandb.group = sweep_id

    # global _cfg
    # _cfg = cfg
    # wandb.agent(sweep_id,
    #             entity=cfg.wandb.entity,
    #             project=cfg.wandb.project,
    #             function=init_run, count=1)

@record
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    cfg.sweep.name = cfg.run
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    local_rank = 0
    device = cfg.device
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")

    if os.environ.get("RANK", "0") == "0":
        init_sweep(cfg)

    if os.environ.get("RANK", "0") == "0":
        cfg.run = generate_run_id_for_sweep(
            f"{cfg.wandb.entity}/{cfg.wandb.project}/{cfg.sweep.id}",
            cfg.sweep.data_dir)
        OmegaConf.save(cfg, os.path.join(f"/tmp/{cfg.sweep.name}.config.yaml"))
        with WandbContext(cfg) as wandb_run:
            wandb_run.tags += (
                f"sweep_id:{cfg.sweep.id}",
                f"sweep_name:{cfg.sweep.name}")
            cfg.device = f'{device}:{local_rank}'
            rollout = MasterSweepRollout(cfg, wandb_run)
            rollout.run()
    else:
        for i in range(10):
            if os.path.exists(f"/tmp/{cfg.sweep.name}.config.yaml"):
                cfg = OmegaConf.load(f"/tmp/{cfg.sweep.name}.config.yaml")
                break
            else:
                logger.debug(f"Waiting for {cfg.sweep.name}.config.yaml to be created")
                time.sleep(10)

        cfg.device = f'{device}:{local_rank}'
        rollout = WorkerSweepRollout(cfg)
        rollout.run()

if __name__ == "__main__":
    main()
