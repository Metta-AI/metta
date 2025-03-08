import logging
import os
import signal  # Aggressively exit on ctrl+c

import hydra
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

def init_run():
    pass

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
    cfg.run = generate_run_id_for_sweep(
        f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}",
        cfg.sweep.data_dir)
    cfg.wandb.group = sweep_id

    wandb.agent(sweep_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                function=init_run, count=1)

@record
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    cfg.sweep.name = cfg.run
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f'{cfg.device}:{local_rank}'
        dist.init_process_group(backend="nccl")

    consecutive_failures = 0
    while True:
        if consecutive_failures > 10:
            logger.debug("Too many consecutive failures, exiting")
            os._exit(0)

        success = False
        try:
            if os.environ.get("RANK", "0") == "0":
                init_sweep(cfg)
                with WandbContext(cfg) as wandb_run:
                    wandb_run.tags += (
                        f"sweep_id:{cfg.sweep.id}",
                        f"sweep_name:{cfg.sweep.name}")
                    rollout = MasterSweepRollout(cfg, wandb_run)
                    success = rollout.run()
            else:
                rollout = WorkerSweepRollout(cfg)
                success = rollout.run()
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1
            raise e

if __name__ == "__main__":
    main()
