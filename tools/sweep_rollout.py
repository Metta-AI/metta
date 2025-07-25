#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import logging
import os
import subprocess
import sys
import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from cogweb.cogweb_client import CogwebClient
from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.protein_utils import apply_protein_suggestion, generate_protein_suggestion
from metta.sweep.wandb_utils import (
    create_wandb_run_for_sweep,
    create_wandb_sweep,
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)

logger = logging.getLogger(__name__)

# Global variable to store original command-line arguments
ORIGINAL_ARGS = []


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    # Store original command-line arguments for later use
    global ORIGINAL_ARGS
    ORIGINAL_ARGS = sys.argv[1:]  # Skip the script name

    logger.info(f"Starting sweep rollout with config: {list(cfg.keys())}")
    logger.debug(f"Full config: {OmegaConf.to_yaml(cfg)}")
    logger.debug(f"Original command-line args: {ORIGINAL_ARGS}")

    # Setup the sweep
    try:
        wandb_sweep_id = setup_sweep(
            cfg.sweep_name,
            cfg.sweep_server_uri,
            cfg.wandb.entity,
            cfg.wandb.project,
            cfg.sweep,
            cfg.runs_dir,
            cfg.sweep_dir,
            logger,
        )
    except Exception as e:
        logger.error(f"Sweep setup failed: {e}", exc_info=True)
        return 1

    # Set the sweep ID in the config
    cfg.sweep_id = wandb_sweep_id
    num_consecutive_failures = 0
    while True:
        err_occurred = False
        # Run the rollout
        try:
            run_single_rollout(cfg)
        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            err_occurred = True
            logger.info(f"Waiting {cfg.rollout_retry_delay} seconds before retry...")
            time.sleep(cfg.rollout_retry_delay)
        if err_occurred:
            num_consecutive_failures += 1
            if num_consecutive_failures > cfg.max_consecutive_failures:
                logger.error(f"Max consecutive failures reached: {cfg.max_consecutive_failures}")
                break
        else:
            num_consecutive_failures = 0

    return 0


def run_single_rollout(cfg: DictConfig) -> int:
    """Run a single rollout."""
    logger.info(f"Starting single rollout for sweep: {cfg.sweep_name}")

    # Validate required config keys
    required_keys = ["sweep", "trainer", "wandb", "sweep_id", "sweep_name", "runs_dir", "sweep_job"]
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        logger.error(f"Available config keys: {list(cfg.keys())}")
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Load previous protein suggestions from WandB
    # and generate next protein suggestion
    protein = MettaProtein(cfg.sweep)
    previous_observation = fetch_protein_observations_from_wandb(
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        wandb_sweep_id=cfg.sweep_id,
        max_observations=cfg.max_observations_to_load,
    )
    logger.info(f"Loaded {len(previous_observation)} previous observations from WandB")
    for obs in previous_observation:
        protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs["is_failure"])
    logger.info(f"Loaded {len(previous_observation)} previous observations from WandB")

    protein_suggestion = generate_protein_suggestion(cfg.trainer, protein)

    # Get next available run name from central DB
    # This enables parallel rollouts across workers
    cogweb_client = CogwebClient.get_client(base_url=cfg.sweep_server_uri)
    run_name = cogweb_client.sweep_client().get_next_run_id(cfg.sweep_name)
    logger.info(f"Got next run name from Cogweb DB: {run_name}")

    # Create a downstream config with the dist_cfg_path set, as well as the sweep specific data_dir
    # We have to do this since the data_dir expected by trainer
    # is the run_dir, which is different from  the sweep's original data_dir.
    downstream_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    assert isinstance(downstream_cfg, DictConfig)
    downstream_cfg.data_dir = f"{cfg.data_dir}/sweep/{cfg.sweep_name}/runs"
    downstream_cfg.dist_cfg_path = f"{cfg.runs_dir}/{run_name}/dist_cfg.yaml"
    downstream_cfg.run = run_name

    # Prepare the run directory.
    os.makedirs(f"{cfg.runs_dir}/{run_name}", exist_ok=True)

    # Create a new run in WandB
    # side-effect: writes dist_cfg.yaml to the run directory
    create_wandb_run_for_sweep(
        wandb_sweep_id=cfg.sweep_id,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        sweep_name=cfg.sweep_name,
        run_name=run_name,
        protein_suggestion=protein_suggestion,
        cfg=downstream_cfg,
    )
    logger.info(f"Created WandB run for sweep: {cfg.sweep_name} with run name: {run_name}")

    # Merge trainer overrides with protein suggestion
    # and save to run directory
    sweep_job_cfg = cfg.sweep_job
    sweep_job_cfg.run = run_name
    apply_protein_suggestion(sweep_job_cfg, protein_suggestion)
    train_cfg_overrides = DictConfig(
        {
            **sweep_job_cfg,
            "run": run_name,
            "run_dir": downstream_cfg.data_dir,
            "sweep_name": cfg.sweep_name,  # Needed by sweep_eval.py
            "wandb": {
                "group": cfg.sweep_name,  # Group all runs under the sweep name
                "name": run_name,  # Individual run name
            },
        }
    )
    trainer_cfg_override_path = os.path.join(downstream_cfg.data_dir, "train_config_overrides.yaml")
    OmegaConf.save(train_cfg_overrides, trainer_cfg_override_path)
    logger.info(f"Wrote trainer overrides to {trainer_cfg_override_path}")

    # Launch trainer as a subprocess and wait for completion
    train_for_run(
        run_name=run_name,
        dist_cfg_path=downstream_cfg.dist_cfg_path,
        data_dir=downstream_cfg.data_dir,
        original_args=ORIGINAL_ARGS,
    )
    logger.info("Launched trainer as a subprocess...")

    # Evaluate the run
    logger.info(f"Evaluating run: {run_name}")

    with WandbContext(downstream_cfg.wandb, downstream_cfg) as wandb_run:
        if wandb_run is None:
            logger.error("Failed to initialize WandB context for evaluation")
            raise RuntimeError("WandB initialization failed during evaluation")

        # Run evaluation
        # side-effect: updates last policy metadata and adds policy to wandb sweep
        eval_results = evaluate_sweep_run(
            wandb_run,
            cfg.sweep.metric,
            cfg.sweep_name,
            downstream_cfg,
        )

        # Record evaluation results in WandB
        wandb_run.summary.update(eval_results)  # type: ignore[attr-defined]
        logger.info(f"Evaluation results: {eval_results}")

        suggestion_cost = eval_results["time.total"]
        suggestion_score = eval_results[cfg.sweep.metric]
        record_protein_observation_to_wandb(
            wandb_run,
            protein_suggestion,
            suggestion_score,
            suggestion_cost,
            False,
        )
        # build obs dictionary for logging purposes only
        obs_dict = {
            "suggestion": protein_suggestion,
            "score": suggestion_score,
            "cost": suggestion_cost,
            "is_failure": False,
        }
        logger.info(f"Recorded protein observation to WandB: {obs_dict}")

        # Save results for all ranks to read
        OmegaConf.save(
            {"eval_metric": suggestion_score, "total_time": suggestion_cost},
            f"{cfg.runs_dir}/{run_name}/sweep_eval_results.yaml",
        )

    return 0


def train_for_run(
    run_name: str,
    dist_cfg_path: str,
    data_dir: str,
    original_args: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """Launch training as a subprocess and wait for completion."""

    # Build the command exactly like the bash script
    cmd = [
        "./devops/train.sh",
        f"run={run_name}",
        f"dist_cfg_path={dist_cfg_path}",
        f"data_dir={data_dir}",
    ]

    # Pass through relevant arguments from the original command line
    # Filter out arguments that we're already setting explicitly
    if original_args:
        skip_prefixes = ["run=", "sweep_name=", "dist_cfg_path=", "data_dir="]
        for arg in original_args:
            # Skip arguments we're already setting
            if any(arg.startswith(prefix) for prefix in skip_prefixes):
                continue
            # Pass through everything else (like hardware configs, wandb settings, etc.)
            cmd.append(arg)

    if logger:
        logger.info(f"[SWEEP:{run_name}] Running: {' '.join(cmd)}")
    else:
        print(f"[SWEEP:{run_name}] Running: {' '.join(cmd)}")

    try:
        # Launch and wait (no capture_output to maintain real-time logging)
        result = subprocess.run(cmd, check=True)
        return result

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[ERROR] Training failed for run: {run_name}")
        else:
            print(f"[ERROR] Training failed for run: {run_name}")
        raise Exception(f"Training failed for {run_name} with exit code {e.returncode}") from e


def evaluate_sweep_run(
    wandb_run: Any,
    sweep_metric: str,
    sweep_name: str,
    global_cfg: DictConfig,
) -> dict[str, Any]:
    simulation_suite_cfg = SimulationSuiteConfig(**OmegaConf.to_container(global_cfg.sim, resolve=True))  # type: ignore[arg-type]
    policy_store = PolicyStore(global_cfg, wandb_run)
    policy_pr = policy_store.policy_record("wandb://run/" + wandb_run.name, selector_type="latest")

    # Load the policy record directly using its wandb URI
    # This will download the artifact and give us a local path
    if not policy_pr.uri:
        raise ValueError(f"Policy record has no URI for run {wandb_run.name}")
    policy_pr = policy_store.load_from_uri(policy_pr.uri)

    eval = SimulationSuite(
        config=simulation_suite_cfg,
        policy_pr=policy_pr,
        policy_store=policy_store,
        device=global_cfg.device,
        vectorization=global_cfg.vectorization,
    )

    # Start evaluation
    eval_start_time = time.time()
    results = eval.simulate()
    eval_time = time.time() - eval_start_time
    eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
    eval_metric = eval_stats_db.get_average_metric_by_filter(sweep_metric, policy_pr)

    # Get training stats from metadata if available
    train_time = policy_pr.metadata.get("train_time", 0)
    agent_step = policy_pr.metadata.get("agent_step", 0)
    epoch = policy_pr.metadata.get("epoch", 0)

    # Update sweep stats with evaluation results
    eval_results = {
        "train.agent_step": agent_step,
        "train.epoch": epoch,
        "time.train": train_time,
        "time.eval": eval_time,
        "time.total": train_time + eval_time,
        "uri": policy_pr.uri,
        "score": eval_metric,
        "score.metric": sweep_metric,
        sweep_metric: eval_metric,  # TODO: Should this be here?
    }

    # Update lineage stats
    for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
        eval_results["lineage." + stat] = eval_results[stat] + policy_pr.metadata.get("lineage." + stat, 0)

    # Update wandb summary
    wandb_run.summary.update(eval_results)

    # Update policy metadata
    policy_pr.metadata.update(
        {
            **eval_results,
            "training_run": wandb_run.name,
        }
    )

    # Add policy to wandb sweep
    policy_store.add_to_wandb_sweep(sweep_name, policy_pr)
    return eval_results


def setup_sweep(
    sweep_name: str,
    sweep_server_uri: str,
    wandb_entity: str,
    wandb_project: str,
    sweep_config: DictConfig,
    runs_dir: str,
    sweep_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Create a new sweep with the given name. If the sweep already exists, skip creation.
    Save the sweep configuration to sweep_dir/metadata.yaml.
    """
    # Check if sweep already exists
    cogweb_client = CogwebClient.get_client(base_url=sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    # Get sweep info and extract wandb_sweep_id if it exists
    sweep_info = sweep_client.get_sweep(sweep_name)
    wandb_sweep_id = sweep_info.wandb_sweep_id if sweep_info.exists else None

    # The sweep hasn't been registered with the centralized DB
    if wandb_sweep_id is None:
        logger.info(f"Creating sweep {sweep_name} in WandB")
        # Create the sweep in WandB with Protein parameters for better visualization
        wandb_sweep_id = create_wandb_sweep(wandb_entity, wandb_project, sweep_name)

        # Write the config for context
        # Save sweep metadata locally
        # in join(cfg.sweep_dir, "metadata.yaml"
        os.makedirs(runs_dir, exist_ok=True)
        OmegaConf.save(
            {
                "sweep": sweep_config,  # The sweep parameters/settings
                "sweep_name": sweep_name,
                "wandb_sweep_id": wandb_sweep_id,
                "wandb_path": f"{wandb_entity}/{wandb_project}/{wandb_sweep_id}",
            },
            os.path.join(sweep_dir, "metadata.yaml"),
        )
        # Register the sweep in the centralized DB
        logger.info(f"Registering sweep {sweep_name} in the centralized DB")
        sweep_client.create_sweep(sweep_name, wandb_project, wandb_entity, wandb_sweep_id)
    else:
        logger.info(f"Found existing sweep {sweep_name} in the centralized DB. WandB sweep ID: {wandb_sweep_id}")

    # Save sweep metadata locally
    # in join(cfg.sweep_dir, "metadata.yaml"
    # Creating runs_dir creates the sweep_dir
    os.makedirs(runs_dir, exist_ok=True)  # TODO: Remove, should not create dir. Should be done by sweep_rollout.
    OmegaConf.save(
        {
            "sweep": sweep_config,  # The sweep parameters/settings
            "sweep_name": sweep_name,
            "wandb_sweep_id": wandb_sweep_id,
            "wandb_path": f"{wandb_entity}/{wandb_project}",
            "wandb_entity": wandb_entity,
            "wandb_project": wandb_project,
        },
        os.path.join(sweep_dir, "metadata.yaml"),
    )
    logger.info(f"Saved sweep metadata to {os.path.join(sweep_dir, 'metadata.yaml')}")
    return wandb_sweep_id


if __name__ == "__main__":
    sys.exit(main())
