#!/usr/bin/env -S uv run

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import logging

import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import json
import os
import time

from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.protein_metta import MettaProtein
from metta.util.metta_script import metta_script

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


def main(cfg: DictConfig) -> int:
    simulation_suite_cfg = SimulationSuiteConfig(**OmegaConf.to_container(cfg.sim, resolve=True))  # type: ignore[arg-type]

    results_path = os.path.join(cfg.run_dir, "sweep_eval_results.yaml")
    start_time = time.time()
    if os.environ.get("NODE_INDEX", "0") != "0":
        while not os.path.exists(results_path):
            time.sleep(1)
            if time.time() - start_time > 500:
                logger.error("Timeout waiting for master to evaluate policy")
                return 1
        return 0

    logger.info(f"Starting evaluation for run: {cfg.run}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        try:
            # Fetch the latest policy record from the run
            policy_pr = policy_store.policy_record("wandb://run/" + cfg.run, selector_type="latest")
            if not policy_pr:
                raise ValueError(f"No policy record found for run {cfg.run}")

            # Load the policy record directly using its wandb URI
            # This will download the artifact and give us a local path
            policy_pr = policy_store.load_from_uri(policy_pr.uri)

        except Exception as e:
            logger.error(f"Error getting policy for run {cfg.run}: {e}")
            # Record failure in WandB directly since we don't have a Protein instance yet
            wandb_run.summary.update({"protein.state": "failure", "protein.error": f"Policy loading failed: {e}"})
            return 1

        eval = SimulationSuite(
            simulation_suite_cfg,
            policy_pr,
            policy_store,
            device=cfg.device,
            vectorization=cfg.vectorization,
        )

        # Start evaluation process
        sweep_stats = {}
        start_time = time.time()

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
        protein_wandb = MettaProtein(cfg.sweep, wandb_run)

        # Record the observation properly so the Protein learns
        protein_wandb.record_observation(eval_metric, total_time)

        # Save results for worker nodes
        if os.environ.get("NODE_INDEX", "0") == "0":
            OmegaConf.save({"eval_metric": eval_metric, "total_time": total_time}, results_path)

        if wandb_run:
            wandb_run.summary.update({"run_time": total_time})
        logger.info(f"Evaluation complete for run: {cfg.run}, score: {eval_metric}")
        return 0


metta_script(main, "sweep_eval")
