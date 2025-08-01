#!/usr/bin/env -S uv run

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import logging
import sys

import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import json
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.util.lock import run_once
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.env_config import create_env_config
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.wandb_utils import record_protein_observation_to_wandb

logger = logging.getLogger(__name__)


def log_file(run_dir, name, data, wandb_run):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        if isinstance(data, DictConfig):
            data = OmegaConf.to_container(data, resolve=True)
        json.dump(data, f, indent=4)

    wandb_run.save(path, base_path=run_dir)


def load_file(run_dir, name):
    path = os.path.join(run_dir, name)
    with open(path, "r") as f:
        return OmegaConf.load(f)


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    simulation_suite_cfg = SimulationSuiteConfig(**OmegaConf.to_container(cfg.sim, resolve=True))  # type: ignore[arg-type]

    # Create env config
    env_cfg = create_env_config(cfg)

    # Load run information from dist_cfg_path
    dist_cfg = OmegaConf.load(cfg.dist_cfg_path)
    logger.info(f"Loaded run info from {cfg.dist_cfg_path}: run={dist_cfg.run}")
    cfg.run = dist_cfg.run
    results_path = os.path.join(cfg.run_dir, "sweep_eval_results.yaml")

    # Function to run evaluation - only executed by rank 0
    def evaluate_policy():
        logger.info(f"Starting evaluation for run: {cfg.run}")

        with WandbContext(cfg.wandb, cfg) as wandb_run:
            policy_store = PolicyStore(cfg, wandb_run)
            try:
                # Fetch the latest policy record from the run
                policy_pr = policy_store.policy_record("wandb://run/" + cfg.run, selector_type="latest")
                if not policy_pr:
                    raise ValueError(f"No policy record found for run {cfg.run}")
                if not policy_pr.uri:
                    raise ValueError(f"Policy record has no URI for run {cfg.run}")

                # Load the policy record directly using its wandb URI
                # This will download the artifact and give us a local path
                policy_pr = policy_store.load_from_uri(policy_pr.uri)

            except Exception as e:
                logger.error(f"Error getting policy for run {cfg.run}: {e}")
                # Record failure in WandB if available
                if wandb_run:
                    wandb_run.summary.update(
                        {"protein.state": "failure", "protein.error": f"Policy loading failed: {e}"}
                    )
                # Save error results for other ranks
                OmegaConf.save({"eval_metric": None, "total_time": 0, "error": str(e)}, results_path)
                return 1

            eval = SimulationSuite(
                simulation_suite_cfg,
                policy_pr,
                policy_store,
                device=cfg.device,
                vectorization=env_cfg.vectorization,
            )

            # Start evaluation process
            sweep_stats = {}

            # Update sweep stats with initial information
            sweep_stats.update(
                {
                    "score.metric": cfg.sweep.metric,
                }
            )
            if wandb_run:
                wandb_run.summary.update(sweep_stats)

            # Start evaluation
            eval_start_time = time.time()

            logger.info(f"Evaluating policy {policy_pr.run_name}")

            if wandb_run:
                log_file(cfg.run_dir, "sweep_eval_config.yaml", cfg, wandb_run)

            results = eval.simulate()
            eval_time = time.time() - eval_start_time
            eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
            eval_metric = eval_stats_db.get_average_metric_by_filter(cfg.sweep.metric, policy_pr)

            # Get training stats from metadata if available
            train_time = policy_pr.metadata.get("train_time", 0)
            agent_step = policy_pr.metadata.get("agent_step", 0)
            epoch = policy_pr.metadata.get("epoch", 0)

            # Update sweep stats with evaluation results
            stats_update = {
                "train.agent_step": agent_step,
                "train.epoch": epoch,
                "time.train": train_time,
                "time.eval": eval_time,
                "time.total": train_time + eval_time,
                "uri": policy_pr.uri,
                "score": eval_metric,
                cfg.sweep.metric: eval_metric,
            }

            sweep_stats.update(stats_update)

            # Update lineage stats
            for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
                sweep_stats["lineage." + stat] = sweep_stats[stat] + policy_pr.metadata.get("lineage." + stat, 0)

            # Update wandb summary
            if wandb_run:
                wandb_run.summary.update(sweep_stats)
            logger.info("Sweep Stats: \n" + json.dumps({k: str(v) for k, v in sweep_stats.items()}, indent=4))

            # Update policy metadata
            policy_pr.metadata.update(
                {
                    **sweep_stats,
                    "training_run": cfg.run,
                }
            )

            # Add policy to wandb sweep
            policy_store.add_to_wandb_sweep(cfg.sweep_name, policy_pr)

            # Record observation in Protein sweep
            total_time = train_time + eval_time

            if wandb_run:
                # Get the protein suggestion that was used for this run
                # TODO: We can use the FS instead of WandB to save an API call.
                protein_suggestion = wandb_run.summary.get("protein_suggestion", {})

                # Record the observation with the actual results
                record_protein_observation_to_wandb(
                    wandb_run,
                    protein_suggestion,  # The suggestion that was evaluated
                    eval_metric or 0,  # The objective value achieved
                    total_time,  # The cost (total time)
                    False,  # is_failure
                )

            # Save results for all ranks to read
            OmegaConf.save({"eval_metric": eval_metric, "total_time": total_time}, results_path)

            if wandb_run:
                wandb_run.summary.update({"run_time": total_time})

            logger.info(f"Evaluation complete for run: {cfg.run}, score: {eval_metric}")

        return 0

    # Use run_once to ensure only rank 0 performs evaluation
    run_once(evaluate_policy)

    # All ranks (including rank 0) wait for results file
    if not os.path.exists(results_path):
        start_time = time.time()
        while not os.path.exists(results_path):
            time.sleep(1)
            if time.time() - start_time > 500:
                logger.error("Timeout waiting for evaluation results")
                return 1

    # Check if evaluation had an error
    results = OmegaConf.load(results_path)
    if "error" in results:
        logger.error(f"Evaluation failed: {results.error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
