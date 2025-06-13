#!/usr/bin/env -S uv run
import os
import sys
from logging import Logger

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.script_decorators import get_metta_logger, metta_script
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.sweep_config_utils import load_train_job_config_with_overrides


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


def train(cfg: ListConfig | DictConfig, wandb_run: WandbRun | None, logger: Logger):
    cfg = load_train_job_config_with_overrides(cfg)

    if os.environ.get("RANK", "0") == "0":
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_job = TrainJob(cfg.train_job)

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = PolicyStore(cfg, wandb_run)
    stats_client: StatsClient | None = get_stats_client(cfg, logger)

    # Instantiate the trainer directly with the typed config
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=train_job.evals,
        stats_client=stats_client,
    )
    trainer.train()
    trainer.close()


@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
@metta_script
@record
def main(cfg: DictConfig) -> int:
    record_heartbeat()

    logger = get_metta_logger()
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

        # Choose appropriate backend based on available hardware
        # TODO: Check if this is still needed
        # I had to add this because I was getting errors when running on CPU
        if (
            torch.cuda.is_available()
            and hasattr(torch.distributed, "is_nccl_available")
            and torch.distributed.is_nccl_available()
        ):
            backend = "nccl"
        else:
            backend = "gloo"

        logger.info(f"Using distributed backend: {backend}")
        dist.init_process_group(backend=backend)

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if os.environ.get("RANK", "0") == "0":
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            train(cfg, wandb_run, logger)
    else:
        train(cfg, None, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
