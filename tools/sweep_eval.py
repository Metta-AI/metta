#!/usr/bin/env -S uv run
import json
import os
import sys
import time

import hydra
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext
from wandb_carbs import WandbCarbs


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
            metta_agent = policy_store.policy("wandb://run/" + cfg.run)
        except Exception as e:
            logger.error(f"Error getting policy for run {cfg.run}: {e}")
            WandbCarbs._record_failure(wandb_run)
            return 1

        eval = SimulationSuite(
            simulation_suite_cfg,
            metta_agent,
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
                "score.metric": cfg.metric,
            }
        )
        wandb_run.summary.update(sweep_stats)

        # Start evaluation
        eval_start_time = time.time()

        logger.info(f"Evaluating policy {metta_agent.name}")
        log_file(cfg.run_dir, "sweep_eval_config.yaml", cfg, wandb_run)

        results = eval.simulate()
        eval_time = time.time() - eval_start_time
        eval_stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
        eval_metric = eval_stats_db.get_average_metric_by_filter(cfg.metric, metta_agent)

        # Get training stats from metadata if available
        train_time = metta_agent.metadata.get("train_time", 0)
        agent_step = metta_agent.metadata.get("agent_step", 0)
        epoch = metta_agent.metadata.get("epoch", 0)

        # Update sweep stats with evaluation results
        stats_update = {
            "train.agent_step": agent_step,
            "train.epoch": epoch,
            "time.train": train_time,
            "time.eval": eval_time,
            "time.total": train_time + eval_time,
            "uri": metta_agent.uri,
            "score": eval_metric,
        }

        sweep_stats.update(stats_update)

        # Update lineage stats
        for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
            sweep_stats["lineage." + stat] = sweep_stats[stat] + metta_agent.metadata.get("lineage." + stat, 0)

        # Update wandb summary
        wandb_run.summary.update(sweep_stats)
        logger.info("Sweep Stats: \n" + json.dumps({k: str(v) for k, v in sweep_stats.items()}, indent=4))

        # Update policy metadata
        metta_agent.metadata.update(
            {
                **sweep_stats,
                "training_run": cfg.run,
            }
        )

        # Add policy to wandb sweep
        policy_store.add_to_wandb_sweep(cfg.sweep_name, metta_agent)

        # Record observation in CARBS if enabled
        total_time = train_time + eval_time
        logger.info(f"Evaluation Metric: {eval_metric}, Total Time: {total_time}")

        WandbCarbs._record_observation(wandb_run, eval_metric, total_time, allow_update=True)

        wandb_run.summary.update({"run_time": total_time})
        return 0


if __name__ == "__main__":
    sys.exit(main())
