import os
import sys
from logging import Logger
from typing import Optional

import hydra
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.sim.map_preview import upload_map_preview
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: Optional[str] = None


def train(cfg, wandb_run, logger: Logger):
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

    trainer = hydra.utils.instantiate(
        cfg.trainer, cfg, wandb_run, policy_store=policy_store, sim_suite_config=train_job.evals
    )
    if train_job.map_preview_uri and trainer.env_cfg._target_ == "metta.env.mettagrid_env.MettaGridEnv":
        # TODO: upload_map_preview() calls MettaGridEnv directly, which will break if our target is MettaGridEnvSet
        # Should we upload a preview for MettaGridEnvSet?
        upload_map_preview(trainer.env_cfg, train_job.map_preview_uri, wandb_run)

    trainer.train()
    trainer.close()


@record
@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
def main(cfg: ListConfig | DictConfig) -> int:
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

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
