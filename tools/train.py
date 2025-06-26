#!/usr/bin/env -S uv run
import os
import sys
from logging import Logger
from typing import Optional

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from app_backend.stats_client import StatsClient
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.util.wandb.wandb_context import WandbContext, WandbRun
from metta.sim.simulation_config import SimulationSuiteConfig


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: Optional[str] = None


def train(cfg: ListConfig | DictConfig, wandb_run: WandbRun | None, logger: Logger):
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

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        cfg.trainer.forward_pass_minibatch_target_size = cfg.trainer.forward_pass_minibatch_target_size // world_size

    stats_client: StatsClient | None = get_stats_client(cfg, logger)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        cfg,
        wandb_run,
        policy_store=policy_store,
        sim_suite_config=train_job.evals,
        stats_client=stats_client,
    )
    trainer.train()
    trainer.close()


@record
@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
def main(cfg: ListConfig | DictConfig) -> int:
    setup_mettagrid_environment(cfg)

    record_heartbeat()

    logger = setup_mettagrid_logger("train")
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
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            train(cfg, wandb_run, logger)
    else:
        train(cfg, None, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
