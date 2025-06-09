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

from metta.rl.protein_opt.metta_protein import MettaProtein
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import config_from_path
from metta.util.lock import run_once
from metta.util.logging import setup_mettagrid_logger
from metta.util.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from metta.util.wandb.wandb_context import WandbContext


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

    # Convert Protein parameter space to WandB format
    logger.info(f"Sweep config: {cfg.sweep}")
    wandb_parameters = _convert_protein_params_to_wandb(cfg.sweep)
    logger.info(f"Converted parameters: {wandb_parameters}")

    sweep_id = wandb.sweep(
        sweep={
            "name": sweep_name,
            "method": "bayes",
            "metric": {"name": "score", "goal": "maximize"},
            "parameters": wandb_parameters,
        },
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
    )
    OmegaConf.save(
        {
            "sweep": sweep_name,
            "wandb_sweep_id": sweep_id,
            "wandb_path": f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}",
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

            protein = MettaProtein(cfg, wandb_run)
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))
            _log_file(
                run_dir,
                wandb_run,
                "protein_state.yaml",
                {
                    "observations": len(protein.success_observations),
                    "failures": len(protein.failure_observations),
                },
            )

            suggestion, _ = protein.suggest()
            logger.info("Generated Protein suggestion: ")
            logger.info(f"\n{'-' * 10}\n{yaml.dump(suggestion, default_flow_style=False)}\n{'-' * 10}")
            _log_file(run_dir, wandb_run, "protein_suggestion.yaml", suggestion)

            train_cfg = OmegaConf.create({key: cfg[key] for key in cfg.sweep.keys()})
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


def apply_protein_suggestion(config: DictConfig | ListConfig, suggestion: dict):
    """Apply suggestions to a configuration object using dotted path notation.

    Args:
        config: The configuration object to modify
        suggestion: The suggestions to apply
    """
    import numpy as np

    def convert_numpy_scalars(obj):
        """Recursively convert numpy scalars to Python primitives."""
        if isinstance(obj, np.number):
            return obj.item()
        elif hasattr(obj, "item") and callable(obj.item):
            try:
                return obj.item()
            except:
                pass
        elif isinstance(obj, dict):
            return {k: convert_numpy_scalars(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_numpy_scalars(item) for item in obj)
        return obj

    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue

        # Convert key to string if it's not already
        str_key = str(key) if not isinstance(key, str) else key

        # Recursively convert numpy scalars to Python primitives
        value = convert_numpy_scalars(value)

        # Use OmegaConf.update with the string key
        OmegaConf.update(config, str_key, value)


def _convert_protein_params_to_wandb(sweep_config: DictConfig | ListConfig) -> dict:
    """Convert Protein parameter space format to WandB sweep parameters format."""
    wandb_params = {}

    # Handle nested sweep structure
    if hasattr(sweep_config, "sweep"):
        if hasattr(sweep_config.sweep, "parameters"):
            # Structure: sweep.parameters.param_name
            params = sweep_config.sweep.parameters
        else:
            # Structure: sweep.param_name (direct parameters under sweep)
            params = sweep_config.sweep
    elif hasattr(sweep_config, "parameters"):
        # Structure: parameters.param_name
        params = sweep_config.parameters
    else:
        # Direct parameter definitions
        params = sweep_config

    for param_name, param_config in params.items():
        if isinstance(param_config, (DictConfig, dict)):
            # Access using dict-style notation for safety
            min_val = param_config.get("min")
            max_val = param_config.get("max")
            distribution = param_config.get("distribution", "uniform")

            if min_val is not None and max_val is not None:
                # Convert string values to numbers if needed
                if isinstance(min_val, str):
                    min_val = float(min_val)
                if isinstance(max_val, str):
                    max_val = float(max_val)

                wandb_param = {"min": min_val, "max": max_val}

                # Convert distribution format
                if distribution == "log_normal":
                    wandb_param["distribution"] = "log_uniform_values"
                elif distribution == "int_uniform":
                    wandb_param["distribution"] = "int_uniform"
                else:
                    wandb_param["distribution"] = "uniform"

                wandb_params[param_name] = wandb_param

    return wandb_params


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
