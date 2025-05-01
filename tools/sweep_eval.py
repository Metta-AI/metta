#!/usr/bin/env python3
import fnmatch
import json
import os
import sys
import time

import hydra
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb_carbs import WandbCarbs

from metta.agent.policy_store import PolicyStore
from metta.eval.analysis_config import AnalyzerConfig
from metta.sim.eval_stats_analyzer import EvalStatsAnalyzer
from metta.sim.eval_stats_db import EvalStatsDB
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


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

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        try:
            policy_pr = policy_store.policy("wandb://run/" + cfg.run)
        except Exception as e:
            logger.error(f"Error getting policy for run {cfg.run}: {e}")
            WandbCarbs._record_failure(wandb_run)
            return 1

        cfg.analyzer.policy_uri = policy_pr.uri

        eval = SimulationSuite(simulation_suite_cfg, policy_pr, policy_store)
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

        logger.info(f"Evaluating policy {policy_pr.name}")
        log_file(cfg.run_dir, "sweep_eval_config.yaml", cfg, wandb_run)

        stats = eval.simulate()
        eval_time = time.time() - eval_start_time

        # Log evaluation stats
        eval_stats_logger = EvalStatsLogger(simulation_suite_cfg, wandb_run)
        eval_stats_logger.log(stats)

        # Create eval stats database and analyze results
        eval_stats_db = EvalStatsDB.from_uri(eval_stats_logger.json_path, cfg.run_dir, wandb_run)

        # Find the metric index in the analyzer metrics
        metric_idxs = [i for i, m in enumerate(cfg.analyzer.analysis.metrics) if fnmatch.fnmatch(cfg.metric, m.metric)]
        if len(metric_idxs) == 0:
            logger.error(f"Metric {cfg.metric} not found in analyzer metrics: {cfg.analyzer.analysis.metrics}")
            return 1
        elif len(metric_idxs) > 1:
            logger.error(f"Multiple metrics found for {cfg.metric} in analyzer")
            return 1
        sweep_metric_index = metric_idxs[0]

        # Analyze the evaluation results
        analyzer_cfg = AnalyzerConfig(cfg.analyzer)
        analyzer = EvalStatsAnalyzer(eval_stats_db, analyzer_cfg.analysis, analyzer_cfg.policy_uri)
        results, _ = analyzer.analyze()

        # Filter by policy name and average the mean values over evals
        filtered_results = results[sweep_metric_index][results[sweep_metric_index]["policy_name"] == policy_pr.name]
        eval_metric = filtered_results[f"mean_{cfg.metric}"].mean()

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
        policy_store.add_to_wandb_sweep(cfg.sweep_name, policy_pr)

        # Record observation in CARBS if enabled
        total_time = train_time + eval_time
        logger.info(f"Evaluation Metric: {eval_metric}, Total Time: {total_time}")

        WandbCarbs._record_observation(wandb_run, eval_metric, total_time, allow_update=True)

        wandb_run.summary.update({"run_time": total_time})
        return 0


if __name__ == "__main__":
    sys.exit(main())
