#!/usr/bin/env -S uv run
import json
import os
import sys
import time
from logging import Logger

import hydra
import wandb
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb_carbs
from metta.common.util.config import config_from_path
from metta.common.util.lock import run_once
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from metta.common.util.wandb.wandb_context import WandbContext
from metta.rl.carbs.metta_carbs import MettaCarbs, carbs_params_from_cfg
from metta.sim.simulation_config import SimulationSuiteConfig


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig | ListConfig) -> int:
    logger = setup_mettagrid_logger("sweep_eval")
    logger.info("Sweep configuration:")
    logger.info(yaml.dump(OmegaConf.to_container(cfg, resolve=True), default_flow_style=False))
    cfg.wandb.name = cfg.sweep_name
    OmegaConf.register_new_resolver("ss", sweep_space, replace=True)
    cfg.sweep = config_from_path(cfg.sweep_params, cfg.sweep_params_override)

    is_master = os.environ.get("NODE_INDEX", "0") == "0"

    run_once(lambda: create_sweep(cfg.sweep_name, cfg, logger))

    if is_master:
        create_run(cfg.sweep_name, cfg, logger)
    else:
        wait_for_run(cfg.sweep_name, cfg, cfg.dist_cfg_path, logger)

    return 0


def create_sweep(sweep_name: str, cfg: DictConfig | ListConfig, logger: Logger) -> None:
    """
    Create a new sweep with the given name.
    """
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.wandb.entity, sweep_name)
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


def create_run(sweep_name: str, cfg: DictConfig | ListConfig, logger: Logger) -> str:
    """
    Create a new run for an existing sweep.
    Returns the run ID.
    """

    sweep_cfg = OmegaConf.load(os.path.join(cfg.sweep_dir, "config.yaml"))

    # Create the simulation suite config to make sure it's valid
    SimulationSuiteConfig(**cfg.sweep_job.evals)

    logger.info(f"Creating new run for sweep: {sweep_name} ({sweep_cfg.wandb_path})")
    run_name = generate_run_id_for_sweep(sweep_cfg.wandb_path, cfg.runs_dir)
    logger.info(f"Sweep run ID: {run_name}")

    run_dir = os.path.join(cfg.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run = run_name
    cfg.run_dir = run_dir

    def init_run():
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            assert wandb_run, "Wandb should be enabled"
            wandb_run_id = wandb_run.id
            wandb_run.name = run_name
            if not wandb_run.tags:
                wandb_run.tags = ()
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
            logger.info(f"\n{'-' * 10}\n{yaml.dump(suggestion, default_flow_style=False)}\n{'-' * 10}")
            _log_file(run_dir, wandb_run, "carbs_suggestion.yaml", suggestion)

            train_cfg = OmegaConf.create({key: cfg[key] for key in cfg.sweep.keys()})
            apply_carbs_suggestion(train_cfg, suggestion)
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


def wait_for_run(sweep_name: str, cfg: DictConfig | ListConfig, path: str, logger: Logger) -> None:
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


def apply_carbs_suggestion(config: DictConfig | ListConfig, suggestion: DictConfig):
    """Apply suggestions to a configuration object using dotted path notation.

    Args:
        config: The configuration object to modify
        suggestion: The suggestions to apply
    """
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue

        # Convert key to string if it's not already
        str_key = str(key) if not isinstance(key, str) else key

        # Use OmegaConf.update with the string key
        OmegaConf.update(config, str_key, value)


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
