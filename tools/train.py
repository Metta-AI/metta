import logging
import os
import sys

import hydra
import torch.distributed as dist
from omegaconf import OmegaConf
from rich.logging import RichHandler
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

# Configure rich colored logging
logging.basicConfig(
    level="INFO", format="%(processName)s %(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("train")


# TODO: populate this more
class TrainJob(Config):
    evals: SimulationSuiteConfig


def train(cfg, wandb_run):
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        logger.info(f"Loading train config overrides from {overrides_path}")
        override_cfg = OmegaConf.load(overrides_path)

        # Set struct flag to False to allow accessing undefined fields
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, override_cfg)
        # Optionally, restore struct behavior after merge
        OmegaConf.set_struct(cfg, True)

    if os.environ.get("RANK", "0") == "0":
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_job = TrainJob(cfg.train_job)

    policy_store = PolicyStore(cfg, wandb_run)
    trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store, train_job.evals)
    trainer.train()
    trainer.close()


@record
@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
def main(cfg: OmegaConf) -> int:
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)
    logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    if "LOCAL_RANK" in os.environ and cfg.device.startswith("cuda"):
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f"{cfg.device}:{local_rank}"
        dist.init_process_group(backend="nccl")

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if os.environ.get("RANK", "0") == "0":
        with WandbContext(cfg, job_type="train") as wandb_run:
            train(cfg, wandb_run)
    else:
        train(cfg, None)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
