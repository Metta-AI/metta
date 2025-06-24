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

    # Load sweep parameters from the specified file
    sweep_params = config_from_path(f"sweep/{cfg.sweep_params}", cfg.sweep_params_override)

    # Merge parameters into the sweep section
    if "parameters" not in cfg.sweep:
        cfg.sweep.parameters = {}
    cfg.sweep.parameters = OmegaConf.merge(cfg.sweep.parameters, sweep_params.parameters)

    is_master = os.environ.get("NODE_INDEX", "0") == "0"

    run_once(lambda: create_sweep(cfg.sweep_name, cfg, logger))

    if is_master:
        create_run(cfg.sweep_name, cfg, logger)
    else:
        wait_for_run(cfg.sweep_name, cfg, cfg.dist_cfg_path, logger)

    return 0


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
            # Access using dict-style notation
            distribution = param_config.get("distribution", "uniform")
            min_val = param_config.get("min")
            max_val = param_config.get("max")

            if distribution == "log_normal":
                wandb_params[param_name] = {"distribution": "log_uniform_values", "min": min_val, "max": max_val}
            elif distribution == "uniform":
                wandb_params[param_name] = {"min": min_val, "max": max_val}
            elif distribution == "int_uniform":
                wandb_params[param_name] = {
                    "min": int(min_val) if min_val is not None else 0,
                    "max": int(max_val) if max_val is not None else 1,
                }

    return wandb_params


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

    # Create WandB sweep with dummy parameter - Protein will override with real suggestions
    logger.info("Creating WandB sweep with dummy parameters for Protein control")

    # Get metric configuration from sweep section
    metric_config = {"name": "protein.objective", "goal": "maximize"}  # Default
    if hasattr(cfg.sweep, "metric"):
        if isinstance(cfg.sweep.metric, str):
            metric_config = {"name": cfg.sweep.metric, "goal": "maximize"}
        elif isinstance(cfg.sweep.metric, (dict, DictConfig)):
            metric_config = OmegaConf.to_container(cfg.sweep.metric, resolve=True)

    sweep_id = wandb.sweep(
        sweep={
            "name": sweep_name,
            "method": "bayes",  # This won't actually be used since Protein overrides suggestions
            "metric": metric_config,
            "parameters": {
                "_protein_dummy": {"values": [1]}  # Dummy parameter - Protein will override all suggestions
            },
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
    SimulationSuiteConfig(**cfg.sweep_job.eval)

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

            protein = MettaProtein(cfg.sweep, wandb_run)
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))
            _log_file(
                run_dir,
                wandb_run,
                "protein_state.yaml",
                {
                    "observations": protein._num_observations,
                    "failures": protein._num_failures,
                    "running": protein._num_running,
                },
            )

            suggestion, _ = protein.suggest()

            logger.info("Generated Protein suggestion: ")
            logger.info(f"\n{'-' * 10}\n{yaml.dump(suggestion, default_flow_style=False)}\n{'-' * 10}")
            _log_file(run_dir, wandb_run, "protein_suggestion.yaml", suggestion)

            # Start with the full configuration (includes all base configs from defaults)
            full_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            # Remove sweep-specific fields that don't belong in training config
            # These fields are for sweep management, not for the actual training run
            sweep_specific_fields = [
                "sweep",  # Contains metric, num_random_samples, parameters
                "sweep_name",
                "sweep_params",
                "sweep_params_override",
                "sweep_dir",
                "runs_dir",
                "sweep_job",  # We'll apply these overrides separately
                "dist_cfg_path",
                "cmd",
            ]
            for field in sweep_specific_fields:
                full_cfg_dict.pop(field, None)

            # Create base training config (has trainer, agent, wandb, etc.)
            train_cfg = OmegaConf.create(full_cfg_dict)

            # Apply sweep_job overrides on top of base config
            # This is where we apply things like trainer.evaluate_interval: 300
            if "sweep_job" in OmegaConf.to_container(cfg, resolve=False):
                sweep_job_overrides = OmegaConf.to_container(cfg.sweep_job, resolve=True)
                for section, overrides in sweep_job_overrides.items():
                    if section != "eval":  # eval is handled separately in train_job
                        OmegaConf.update(train_cfg, section, overrides)

            logger.info(f"Protein suggestions: {suggestion}")
            # Suggestion is already cleaned by MettaProtein._transform_suggestion()
            suggestion_config = OmegaConf.create(suggestion)
            if isinstance(suggestion_config, DictConfig):
                apply_protein_suggestion(train_cfg, suggestion_config)
            else:
                logger.error(f"Unexpected suggestion config type: {type(suggestion_config)}")
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


def apply_protein_suggestion(config: DictConfig | ListConfig, suggestion: DictConfig):
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
