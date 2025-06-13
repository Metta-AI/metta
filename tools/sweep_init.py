#!/usr/bin/env -S uv run

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import json
import os
import sys
import time
from logging import Logger

import hydra
import wandb
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.lock import run_once
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from metta.common.util.wandb.wandb_context import WandbContext
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.protein_wandb import create_sweep as create_wandb_sweep


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
@metta_script
def main(cfg: DictConfig | ListConfig) -> int:
    # TODO: Check: logger should be sweep_init?
    logger = setup_mettagrid_logger("sweep_eval")

    # Extract sweep base name from CLI sweep_run parameter (e.g., "simple_sweep")
    # Individual training runs will be "simple_sweep.r.0", etc.
    cfg.sweep_job.sweep_run = cfg.sweep_run  # Store in sweep_job for other scripts to use

    cfg.wandb.name = cfg.sweep_run

    is_master = os.environ.get("NODE_INDEX", "0") == "0"

    run_once(lambda: create_sweep(cfg, logger))

    if is_master:
        create_run(cfg, logger)
    else:
        wait_for_run(cfg, cfg.dist_cfg_path, logger)

    return 0


def create_sweep(cfg: DictConfig | ListConfig, logger: Logger) -> None:
    """
    Create a new sweep with the given name.
    """
    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.wandb.entity, cfg.sweep_run)
    if sweep_id is not None:
        logger.info(f"Sweep already exists, skipping creation for: {cfg.sweep_run}")
        return

    logger.info(f"Creating new sweep: {cfg.sweep_dir}")
    os.makedirs(cfg.runs_dir, exist_ok=True)

    # Create sweep using the standalone function (Protein will control all parameters)
    sweep_id = create_wandb_sweep(cfg.sweep_run, cfg.wandb.entity, cfg.wandb.project)
    OmegaConf.save(
        {
            "sweep": cfg.sweep_run,
            "wandb_sweep_id": sweep_id,
            "wandb_path": f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}",
        },
        os.path.join(cfg.sweep_dir, "config.yaml"),
    )


def create_run(cfg: DictConfig | ListConfig, logger: Logger) -> str:
    """
    Create a new run for an existing sweep.
    Returns the run ID.
    """

    sweep_cfg = OmegaConf.load(os.path.join(cfg.sweep_dir, "config.yaml"))

    # Load eval config to make sure it's valid.
    # TODO: Remove since pydantic should validate it anyways.
    eval_config = cfg.sweep_job.evals
    SimulationSuiteConfig(**eval_config)

    run_name = generate_run_id_for_sweep(sweep_cfg.wandb_path, cfg.runs_dir)
    logger.info(f"Creating new run for sweep: {cfg.sweep_run} ({sweep_cfg.wandb_path})")
    logger.info(f"Sweep run ID: {run_name}")

    run_dir = os.path.join(cfg.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run = run_name  # Top-level for training scripts
    cfg.run_dir = run_dir  # Top-level for training scripts

    # Now log the resolved configuration
    logger.info("Sweep configuration (resolved):")
    logger.info(yaml.dump(OmegaConf.to_container(cfg, resolve=True), default_flow_style=False))

    def init_run():
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            assert wandb_run, "Wandb should be enabled"
            wandb_run_id = wandb_run.id
            wandb_run.name = run_name
            if not wandb_run.tags:
                wandb_run.tags = ()
            wandb_run.tags += (f"sweep_id:{sweep_cfg.wandb_sweep_id}", f"sweep_run:{sweep_cfg.sweep}")

            protein = MettaProtein(cfg.sweep_job.sweep, wandb_run)
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))

            # Log protein state - fix AttributeErrors
            protein_state = {
                "num_observations": len(protein._observations),
                "observations": protein._observations,
                "hyperparameters": {
                    name: {"min": space.min, "max": space.max, "scale": space.scale, "search_center": space.mean}
                    for name, space in protein._protein.hyperparameters.flat_spaces.items()
                },
            }
            _log_file(run_dir, wandb_run, "protein_state.yaml", protein_state)

            suggestion, info = protein.suggest()
            logger.info("Generated Protein suggestion: ")
            logger.info(f"\n{'-' * 10}\n{yaml.dump(suggestion, default_flow_style=False)}\n{'-' * 10}")
            if info:
                logger.info(f"Suggestion metadata: {info}")
            _log_file(run_dir, wandb_run, "protein_suggestion.yaml", suggestion)

            # TODO: Add Pydantic validation to ensure no parameter overlap between sweep and sweep_job

            logger.info(f"train_cfg: {cfg.sweep_job.trainer}")

            # Apply Protein suggestions on top of sweep_job overrides
            logger.info("=== APPLYING PROTEIN SUGGESTIONS ===")
            apply_protein_suggestion(cfg.sweep_job, suggestion)

            save_path = os.path.join(run_dir, "train_config_overrides.yaml")

            # TODO: Only the configs exposed in sweep_job are saved.
            # Is this actually what we want?
            # Save with resolve=True to resolve all interpolations
            OmegaConf.save(OmegaConf.to_container(cfg.sweep_job, resolve=True), save_path)

            # TODO: Should we be saving the whole config?
            logger.info(f"Saved train config overrides to {save_path}")
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


def wait_for_run(cfg: DictConfig | ListConfig, path: str, logger: Logger) -> None:
    """
    Wait for a run to exist.
    """
    for _ in range(10):
        if os.path.exists(path):
            break
        logger.info(f"Waiting for run for sweep: {cfg.sweep_run}")
        time.sleep(5)

    run = OmegaConf.load(path).run
    logger.info(f"Run ID: {run} ready")


def apply_protein_suggestion(config: DictConfig | ListConfig, suggestion: DictConfig):
    """Apply suggestions to a configuration object using deep merge.

    Args:
        config: The configuration object to modify
        suggestion: The suggestions to apply
    """
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue

        # For nested structures, merge instead of overwrite
        if key in config and isinstance(config[key], DictConfig) and isinstance(value, dict):
            # Deep merge for nested configs
            config[key] = OmegaConf.merge(config[key], value)
        else:
            # Direct assignment for non-nested values
            config[key] = value


def _log_file(run_dir: str, wandb_run, name: str, data):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        if isinstance(data, DictConfig):
            data = OmegaConf.to_container(data, resolve=False)

        # Convert data to JSON-serializable format to handle WandB objects
        try:
            json.dump(data, f, indent=4)
        except TypeError:
            # Handle non-serializable objects (like SummarySubDict) by converting to strings
            serializable_data = json.loads(json.dumps(data, default=str))
            json.dump(serializable_data, f, indent=4)

    wandb_run.save(path, base_path=run_dir)


if __name__ == "__main__":
    sys.exit(main())
