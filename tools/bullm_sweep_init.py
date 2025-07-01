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

from metta.common.util.config import config_from_path
from metta.common.util.lock import run_once
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.script_decorators import metta_script
from metta.common.util.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from metta.common.util.wandb.wandb_context import WandbContext
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.protein_wandb import create_sweep as wandb_create_sweep


@hydra.main(config_path="../configs/sweep", config_name="bullm_sweep", version_base=None)
@metta_script
def main(cfg: DictConfig | ListConfig) -> int:
    logger = setup_mettagrid_logger("sweep_eval")
    logger.info("Sweep configuration:")
    logger.info(yaml.dump(OmegaConf.to_container(cfg, resolve=True), default_flow_style=False))
    cfg.wandb.name = cfg.run
    OmegaConf.register_new_resolver("ss", sweep_space, replace=True)

    is_master = os.environ.get("NODE_INDEX", "0") == "0"

    run_once(lambda: init_sweep(cfg.run, cfg, logger))

    if is_master:
        create_run(cfg.run, cfg, logger)
    else:
        wait_for_run(cfg.run, cfg, cfg.dist_cfg_path, logger)

    return 0


def init_sweep(sweep_name: str, cfg: DictConfig | ListConfig, logger: Logger) -> None:
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.wandb.entity, sweep_name)
    if sweep_id is not None:
        logger.info(f"Sweep already exists, skipping creation for: {sweep_name}")
        return

    logger.info(f"Creating new sweep: {cfg.run_dir}")
    os.makedirs(cfg.run_dir, exist_ok=True)

    sweep_id = wandb_create_sweep(sweep_name, cfg.wandb.entity, cfg.wandb.project)
    OmegaConf.save(
        {
            "sweep": sweep_name,
            "wandb_sweep_id": sweep_id,
            "wandb_path": f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}",
        },
        os.path.join(cfg.run_dir, "config.yaml"),
    )


def create_run(sweep_name: str, cfg: DictConfig | ListConfig, logger: Logger) -> str:
    sweep_cfg = OmegaConf.load(os.path.join(cfg.run_dir, "config.yaml"))
    SimulationSuiteConfig(**cfg.eval)

    logger.info(f"Creating new run for sweep: {sweep_name} ({sweep_cfg.wandb_path})")
    run_name = generate_run_id_for_sweep(sweep_cfg.wandb_path, cfg.run_dir)
    logger.info(f"Sweep run ID: {run_name}")

    run_dir = os.path.join(cfg.run_dir, run_name)
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

            protein = MettaProtein(cfg, wandb_run)
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))
            _log_file(
                run_dir,
                wandb_run,
                "protein_state.yaml",
                {
                    "generation": protein.generation,
                    "observations": protein._observations,
                    "params": str(protein._protein.params),
                },
            )

            suggestion = protein.suggest()
            logger.info("Generated Protein suggestion: ")
            logger.info(f"\n{'-' * 10}\n{yaml.dump(suggestion, default_flow_style=False)}\n{'-' * 10}")
            _log_file(run_dir, wandb_run, "protein_suggestion.yaml", suggestion)

            train_cfg = OmegaConf.create({key: cfg[key] for key in cfg.parameters.keys()})
            apply_protein_suggestion(train_cfg, suggestion)
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
    for _ in range(10):
        if os.path.exists(path):
            break
        logger.info(f"Waiting for run for sweep: {sweep_name}")
        time.sleep(5)

    run = OmegaConf.load(path).run
    logger.info(f"Run ID: {run} ready")


def apply_protein_suggestion(config: DictConfig | ListConfig, suggestion: DictConfig):
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        str_key = str(key) if not isinstance(key, str) else key
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
