#!/usr/bin/env python3
import json
import logging
import os
import sys
import time

import hydra
import wandb
import wandb_carbs
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from metta.rl.carbs.metta_carbs import MettaCarbs, carbs_params_from_cfg
from metta.rl.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from metta.rl.wandb.wandb_context import WandbContext
from metta.util.config import config_from_path
from metta.util.efs_lock import efs_lock

# Configure rich colored logging to stderr instead of stdout
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])

logger = logging.getLogger("sweep_init")


@hydra.main(config_path="../configs", config_name="sweep", version_base=None)
def main(cfg: OmegaConf) -> int:
    cfg.wandb.name = cfg.sweep_name
    OmegaConf.register_new_resolver("ss", sweep_space, replace=True)
    cfg.sweep = config_from_path(cfg.sweep_params, cfg.sweep_params_override)

    is_master = os.environ.get("NODE_INDEX", "0") == "0"
    if is_master:
        with efs_lock(cfg.sweep_dir + "/lock", timeout=300):
            create_sweep(cfg.sweep_name, cfg)
        create_run(cfg.sweep_name, cfg)
    else:
        wait_for_run(cfg.sweep_name, cfg, cfg.dist_cfg_path)


def create_sweep(sweep_name: str, cfg: OmegaConf) -> None:
    """
    Create a new sweep with the given name.
    """
    sweep_id = sweep_id_from_name(cfg.wandb.project, sweep_name)
    if sweep_id is not None:
        logger.info(f"Sweep already exists, skipping creation for: {sweep_name}")
        return

    logger.info(f"Creating new sweep: {cfg.sweep_dir}")
    os.makedirs(cfg.runs_dir, exist_ok=True)

    carbs_params = carbs_params_from_cfg(cfg)
    sweep_id = wandb_carbs.create_sweep(sweep_name, cfg.wandb.entity, cfg.wandb.project, carbs_params[0])
    OmegaConf.save(
        {
            "sweep": sweep_name,
            "wandb_sweep_id": sweep_id,
            "wandb_path": f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}",
            "carbs_params": carbs_params[0],
        },
        os.path.join(cfg.sweep_dir, "config.yaml"),
    )


def create_run(sweep_name: str, cfg: OmegaConf) -> str:
    """
    Create a new run for an existing sweep.
    Returns the run ID.
    """

    sweep_cfg = OmegaConf.load(os.path.join(cfg.sweep_dir, "config.yaml"))

    logger.info(f"Creating new run for sweep: {sweep_name} ({sweep_cfg.wandb_path})")
    run_name = generate_run_id_for_sweep(sweep_cfg.wandb_path, cfg.runs_dir)
    logger.info(f"Sweep run ID: {run_name}")

    run_dir = os.path.join(cfg.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run = run_name

    def init_run():
        with WandbContext(cfg, data_dir=run_dir) as wandb_run:
            wandb_run_id = wandb_run.id
            wandb_run.name = run_name
            wandb_run.tags += (f"sweep_id:{sweep_cfg.wandb_sweep_id}", f"sweep_name:{sweep_cfg.sweep}")

            carbs = MettaCarbs(cfg, wandb_run)
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))
            _log_file(
                run_dir,
                wandb_run,
                "carbs_state.yaml",
                {
                    "generation": carbs.generation,
                    "observations": carbs._observations,
                    "params": str(carbs._carbs.params),
                },
            )

            suggestion = carbs.suggest()
            logger.info("Generated CARBS suggestion: ")
            logger.info("\n" + "-" * 10 + "\n" + yaml.dump(suggestion, default_flow_style=False) + "\n" + "-" * 10)
            _log_file(run_dir, wandb_run, "carbs_suggestion.yaml", suggestion)

            train_cfg = OmegaConf.create({key: cfg[key] for key in cfg.sweep.keys()})
            _apply_carbs_suggestion(train_cfg, suggestion)
            save_path = os.path.join(run_dir, "train_config_overrides.yaml")
            OmegaConf.save(train_cfg, save_path)
            logger.info(f"Saved train config overrides to {save_path}")

        if cfg.dist_cfg_path is not None:
            logger.info(f"Saved run details to {cfg.dist_cfg_path}")
            os.makedirs(os.path.dirname(cfg.dist_cfg_path), exist_ok=True)
            OmegaConf.save(
                {
                    "run": run_name,
                    "wandb_run_id": wandb_run_id,
                },
                cfg.dist_cfg_path,
            )

    wandb.agent(
        sweep_cfg.wandb_sweep_id, entity=cfg.wandb.entity, project=cfg.wandb.project, function=init_run, count=1
    )

    return run_name


def wait_for_run(sweep_name: str, cfg: OmegaConf, path: str) -> None:
    """
    Wait for a run to exist.
    """
    for _ in range(10):
        if os.path.exists(path):
            break
        logger.info(f"Waiting for run for sweep: {sweep_name}")
        time.sleep(5)

    run = OmegaConf.load(path).run
    logger.info(f"Run ID: {run} ready")


def _apply_carbs_suggestion(config: OmegaConf, suggestion: DictConfig):
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        new_cfg_param = config
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            if k in new_cfg_param:
                new_cfg_param = new_cfg_param[k]
            else:
                new_cfg_param[k] = {}
                new_cfg_param = new_cfg_param[k]
        param_name = key_parts[-1]
        new_cfg_param[param_name] = value


def _log_file(run_dir: str, wandb_run, name: str, data):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        if isinstance(data, DictConfig):
            data = OmegaConf.to_container(data, resolve=False)
        json.dump(data, f, indent=4)

    wandb_run.save(path, base_path=run_dir)


def sweep_space(space, min_val, max_val, center=None, *, _root_):
    result = {
        "space": space,
        "min": min_val,
        "max": max_val,
        "search_center": center,
    }
    if space == "int":
        result["is_int"] = True
        result["space"] = "linear"
    return OmegaConf.create(result)


if __name__ == "__main__":
    sys.exit(main())
