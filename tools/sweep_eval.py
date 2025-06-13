#!/usr/bin/env -S uv run

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import json
import os
import sys
import time

import hydra
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.common.util.script_decorators import metta_script
from metta.common.util.wandb.wandb_context import WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sweep.protein_metta import MettaProtein


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
@metta_script
def main(cfg: DictConfig | ListConfig) -> int:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("sweep_eval")

    logger.info("Sweep configuration:")
    logger.info(yaml.dump(OmegaConf.to_container(cfg, resolve=True), default_flow_style=False))
    simulation_suite_cfg = SimulationSuiteConfig(**cfg.sweep_job.evals)

    results_path = os.path.join(cfg.run_dir, "sweep_eval_results.yaml")
    start_time = time.time()
    if os.environ.get("NODE_INDEX", "0") != "0":
        logger.info("Waiting for master to evaluate policy")
        while not os.path.exists(results_path):
            time.sleep(1)
            if time.time() - start_time > 500:
                logger.error("Timeout waiting for master to evaluate policy")
                return 1
        return 0

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        try:
            # First get the policy records from the run to find the artifact URI
            policy_records = policy_store.policy_records("wandb://run/" + cfg.run)
            if not policy_records:
                raise ValueError(f"No policy records found for run {cfg.run}")

            # Get the first policy record and load it directly using its wandb URI
            # This will download the artifact and give us a local path
            policy_pr = policy_store.load_from_uri(policy_records[0].uri)
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
                "score.metric": cfg.sweep_job.sweep.parameters.metric,
            }
        )
        wandb_run.summary.update(sweep_stats)

        # Start evaluation
        eval_start_time = time.time()

        logger.info(f"Evaluating policy {policy_pr.name}")
        log_file(cfg.run_dir, "sweep_eval_config.yaml", cfg, wandb_run)

        results = eval.simulate()
        eval_time = time.time() - eval_start_time
        eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
        eval_metric = eval_stats_db.get_average_metric_by_filter(cfg.sweep_job.sweep.parameters.metric, policy_pr)

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
        policy_store.add_to_wandb_sweep(cfg.sweep_run, policy_pr)

        # Record observation in Protein sweep
        total_time = train_time + eval_time
        logger.info(f"Evaluation Metric: {eval_metric}, Total Time: {total_time}")

        try:
            # Recreate the WandbProtein instance to properly record observations
            # Create the protein with the sweep configuration
            if hasattr(cfg.sweep_job, "sweep") and cfg.sweep_job.sweep:
                protein_wandb = MettaProtein(cfg.sweep_job.sweep, wandb_run)

                # Record the observation properly so the Protein learns
                if eval_metric is None:
                    logger.warning("Evaluation metric is None - recording as failure")
                    protein_wandb.record_failure("Evaluation failed")
                else:
                    logger.info(f"Recording observation: objective={eval_metric}, cost={total_time}")
                    protein_wandb.record_observation(eval_metric, total_time)
            else:
                logger.info("No sweep configuration found - not a Protein sweep")
                wandb_run.summary.update(
                    {
                        "protein.objective": eval_metric or 0.0,
                        "protein.cost": total_time,
                        "protein.state": "failure" if eval_metric is None else "success",
                    }
                )

        except AttributeError as e:
            # Missing sweep config - this run is not part of a Protein sweep
            logger.info(f"Not a Protein sweep (missing config): {e}")
            wandb_run.summary.update(
                {
                    "protein.objective": eval_metric or 0.0,
                    "protein.cost": total_time,
                    "protein.state": "failure" if eval_metric is None else "success",
                }
            )
        except Exception as e:
            # Network issues, WandB API failures, etc.
            logger.warning(f"Protein recording failed (network/API issue): {e}")
            # Fallback to direct WandB update
            wandb_run.summary.update(
                {
                    "protein.objective": eval_metric or 0.0,
                    "protein.cost": total_time,
                    "protein.state": "failure" if eval_metric is None else "success",
                }
            )

        wandb_run.summary.update({"run_time": total_time})
        return 0


if __name__ == "__main__":
    sys.exit(main())
