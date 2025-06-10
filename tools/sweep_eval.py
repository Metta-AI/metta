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
            agent = policy_store.policy("wandb://run/" + cfg.run)
        except Exception as e:
            logger.error(f"Error getting policy for run {cfg.run}: {e}")
            WandbCarbs._record_failure(wandb_run)
            return 1

        sim = SimulationSuite(
            simulation_suite_cfg,
            policy_agent=agent,
            policy_store=policy_store,
            device=cfg.device,
            vectorization=cfg.vectorization,
        )

        logger.info(f"Starting simulation for {agent.name}")
        start_time = time.time()
        sim_results = sim.simulate()
        end_time = time.time()

        logger.info(f"Evaluating policy {agent.name}")
        eval_stats_db = EvalStatsDB.from_sim_stats_db(sim_results.stats_db)

        # Calculate core metrics
        eval_metric = eval_stats_db.get_average_metric_by_filter(cfg.metric, agent)
        train_time = agent.metadata.get("train_time", 0)
        agent_step = agent.metadata.get("agent_step", 0)
        epoch = agent.metadata.get("epoch", 0)

        # Construct sweep stats with all relevant metrics
        sweep_stats = {
            "eval_time": end_time - start_time,
            "total_envs": 0,  # Legacy field, now unused
            "train_time": train_time,
            "agent_step": agent_step,
            "epoch": epoch,
            "uri": agent.uri,
            cfg.metric: eval_metric,
        }

        # Add lineage stats if available
        for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
            sweep_stats["lineage." + stat] = sweep_stats[stat] + agent.metadata.get("lineage." + stat, 0)

        # Update the agent metadata with the new eval metric and sweep stats
        # This enables tracking of evaluation results over time
        agent.metadata.update(
            {
                **sweep_stats,
                "eval_scores": {cfg.metric: eval_metric},
            }
        )

        policy_store.add_to_wandb_sweep(cfg.sweep_name, agent)

        # Record observation in CARBS if enabled
        total_time = train_time + (end_time - start_time)
        logger.info(f"Evaluation Metric: {eval_metric}, Total Time: {total_time}")

        WandbCarbs._record_observation(wandb_run, eval_metric, total_time, allow_update=True)

        wandb_run.summary.update({"run_time": total_time})
        return 0


if __name__ == "__main__":
    sys.exit(main())
