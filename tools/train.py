#!/usr/bin/env -S uv run

import logging
import os
import platform
from datetime import datetime
from logging import Logger

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.git import get_git_hash_for_remote_task
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.core.distributed import setup_device_and_distributed
from metta.rl.system_config import create_system_config
from metta.rl.trainer import train
from metta.rl.trainer_config import create_trainer_config
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.metta_script import metta_script
from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    validate_train_job_config,
)
from tools.utils import calculate_default_num_workers, get_policy_store_from_cfg

logger = logging.getLogger(__name__)


# TODO: populate this more
class TrainJob(Config):
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


def handle_train(cfg: DictConfig, wandb_run: WandbRun | None, logger: Logger):
    cfg = load_train_job_config_with_overrides(cfg)

    # Create env config early to use it throughout
    system_cfg = create_system_config(cfg)

    # Validation must be done after merging
    # otherwise trainer's default num_workers: null will be override the values
    # set by _calculate_default_num_workers, and the validation will fail
    if not cfg.trainer.num_workers:
        cfg.trainer.num_workers = calculate_default_num_workers(system_cfg.vectorization == "serial")

    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

    # Determine git hash for remote simulations
    if cfg.trainer.simulation.evaluate_remote:
        if not stats_client:
            cfg.trainer.simulation.evaluate_remote = False
            logger.info("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.simulation.evaluate_interval:
            cfg.trainer.simulation.evaluate_remote = False
            logger.info("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.simulation.git_hash:
            cfg.trainer.simulation.git_hash = get_git_hash_for_remote_task(
                skip_git_check=cfg.trainer.simulation.skip_git_check,
                skip_cmd="trainer.simulation.skip_git_check=true",
                logger=logger,
            )
            if cfg.trainer.simulation.git_hash:
                logger.info(f"Git hash for remote evaluations: {cfg.trainer.simulation.git_hash}")
            else:
                logger.info("No git hash available for remote evaluations")

    cfg = validate_train_job_config(cfg)

    if os.environ.get("RANK", "0") == "0":  # master only
        logger.info("Trainer config after overrides:\n%s", OmegaConf.to_yaml(cfg.trainer, resolve=True))
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_job = TrainJob.model_validate(OmegaConf.to_container(cfg.train_job, resolve=True))
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = get_policy_store_from_cfg(cfg, wandb_run)

    # Use the functional train interface directly
    train(
        run=cfg.run,
        run_dir=cfg.run_dir,
        system_cfg=system_cfg,
        agent_cfg=cfg.agent,
        device=torch.device(system_cfg.device),
        trainer_cfg=create_trainer_config(cfg),
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=train_job.evals,
        stats_client=stats_client,
    )


def _set_min(cfg: DictConfig, path: str, value: float) -> None:
    """Set config value to the minimum of the current value and the provided value.
    If current value is None, use the provided value."""
    current = OmegaConf.select(cfg, path)
    if current is None:
        OmegaConf.update(cfg, path, value)
    else:
        OmegaConf.update(cfg, path, min(value, current))


def apply_mac_overrides(cfg: DictConfig) -> None:
    if not cfg.bypass_mac_overrides and platform.system() == "Darwin":
        _set_min(cfg, "trainer.batch_size", 1024)
        _set_min(cfg, "trainer.minibatch_size", 1024)
        _set_min(cfg, "trainer.forward_pass_minibatch_target_size", 2)
        _set_min(cfg, "trainer.checkpoint.checkpoint_interval", 10)
        _set_min(cfg, "trainer.checkpoint.wandb_checkpoint_interval", 10)
        _set_min(cfg, "trainer.bptt_horizon", 8)
        _set_min(cfg, "trainer.simulation.evaluate_interval", 10)


def set_run_name_if_missing(cfg: DictConfig) -> None:
    """Set up cfg.run if it's not already set."""
    if (OmegaConf.is_missing(cfg, "run") or not cfg.get("run")) and cfg.run_name_pattern:
        generated_name = cfg.run_name_pattern
        replacements = {
            "user": os.getenv("USER", "unknown_user"),
            "now": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "curriculum": cfg.trainer.curriculum.split("/")[-1],
        }
        for key, replacement in replacements.items():
            generated_name = generated_name.replace(f"{{{key}}}", replacement)
        if not all(c.isalnum() or c in [".", "_", "-"] for c in generated_name):
            raise ValueError(f"Invalid run name pattern: {cfg.run_name_pattern} -> {generated_name}")
        print(f"Setting run name to {generated_name}")
        cfg.run = generated_name


@record
def main(cfg: DictConfig) -> int:
    record_heartbeat()

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    apply_mac_overrides(cfg)
    # Use shared distributed setup function
    device, is_master, world_size, rank = setup_device_and_distributed(cfg.device)

    # Update cfg.device to include the local rank if distributed
    cfg.device = str(device)

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if is_master:
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            handle_train(cfg, wandb_run, logger)
    else:
        handle_train(cfg, None, logger)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return 0


metta_script(main, config_name="train_job", pre_main=set_run_name_if_missing)
