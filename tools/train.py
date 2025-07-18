#!/usr/bin/env -S uv run
import logging
import multiprocessing
import os
from logging import Logger

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.mettagrid.curriculum.core import Curriculum
from metta.rl.curriculum.curriculum_server import CurriculumServer
from metta.rl.trainer import MettaTrainer
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.metta_script import metta_script
from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    validate_train_job_config,
)

logger = logging.getLogger(__name__)


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


def _calculate_default_num_workers(is_serial: bool) -> int:
    if is_serial:
        return 1

    cpu_count = multiprocessing.cpu_count() or 1

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    ideal_workers = (cpu_count // 2) // num_gpus

    # Round down to nearest power of 2
    num_workers = 1
    while num_workers * 2 <= ideal_workers:
        num_workers *= 2

    return max(1, num_workers)


def train(cfg: DictConfig, wandb_run: WandbRun | None, logger: Logger, curriculum: Curriculum | None):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = PolicyStore(cfg, wandb_run)  # type: ignore[reportArgumentType]
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

    train_job = TrainJob(cfg.train_job)

    trainer = MettaTrainer(
        cfg,
        curriculum,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=train_job.evals,
        stats_client=stats_client,
    )

    try:
        trainer.train()
    finally:
        trainer.close()


@record
def main(cfg: DictConfig) -> int:
    record_heartbeat()

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    cfg = load_train_job_config_with_overrides(cfg)

    # Validation must be done after merging
    # otherwise trainer's default num_workers: null will be override the values
    # set by _calculate_default_num_workers, and the validation will fail
    if not cfg.trainer.num_workers:
        cfg.trainer.num_workers = _calculate_default_num_workers(cfg.vectorization == "serial")

    cfg = validate_train_job_config(cfg)

    if "LOCAL_RANK" in os.environ and cfg.device.startswith("cuda"):
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f"{cfg.device}:{local_rank}"
        dist.init_process_group(backend="nccl")

    logger.info(f"Training {cfg.run} on {cfg.device}")

    curriculum_server = None
    if os.environ.get("RANK", "0") == "0":
        logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        curriculum_server = CurriculumServer.create(cfg.trainer)

        with WandbContext(cfg.wandb, cfg) as wandb_run:
            train(cfg, wandb_run, logger, curriculum_server)
    else:
        train(cfg, None, logger, None)

    if dist.is_initialized():
        if curriculum_server is not None:
            curriculum_server.stop()
        dist.destroy_process_group()

    return 0


metta_script(main, "train_job")
