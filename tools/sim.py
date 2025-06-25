#!/usr/bin/env -S uv run
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from app_backend.stats_client import StatsClient
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.common.util.stats_client_cfg import get_stats_client
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimJob(Config):
    __init__ = Config.__init__
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    stats_db_uri: str
    stats_dir: str  # The (local) directory where stats should be stored
    replay_dir: str  # where to store replays


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def simulate_policy(
    sim_job: SimJob,
    policy_uri: str,
    cfg: DictConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.

    Returns:
        Dictionary containing simulation results and metrics
    """
    results = {"policy_uri": policy_uri, "checkpoints": []}

    policy_store = PolicyStore(cfg, None)
    # TODO: institutionalize this better?
    metric = sim_job.simulation_suite.name + "_score"
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type, n=1, metric=metric)

    stats_client: StatsClient | None = get_stats_client(cfg, logger)

    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")
        replay_dir = f"{sim_job.replay_dir}/{pr.name}"
        sim = SimulationSuite(
            config=sim_job.simulation_suite,
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=replay_dir,
            stats_dir=sim_job.stats_dir,
            device=cfg.device,
            vectorization=cfg.vectorization,
            stats_client=stats_client,
        )
        sim_results = sim.simulate()

        # Collect metrics from the results
        checkpoint_data = {"name": pr.name, "uri": pr.uri, "metrics": {}}

        # Get average reward
        rewards_df = sim_results.stats_db.query(
            "SELECT AVG(value) AS reward_avg FROM agent_metrics WHERE metric = 'reward'"
        )
        if len(rewards_df) > 0 and rewards_df.iloc[0]["reward_avg"] is not None:
            checkpoint_data["metrics"]["reward_avg"] = float(rewards_df.iloc[0]["reward_avg"])

        results["checkpoints"].append(checkpoint_data)

        # Export the stats DB
        logger.info("Exporting merged stats DB → %s", sim_job.stats_db_uri)
        sim_results.stats_db.export(sim_job.stats_db_uri)

        logger.info("Evaluation complete for policy %s", pr.uri)

    return results


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.sim")
    logger.info(f"Sim job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    sim_job = SimJob(cfg.sim_job)

    all_results = {"simulation_suite": sim_job.simulation_suite.name, "policies": []}

    for policy_uri in sim_job.policy_uris:
        policy_results = simulate_policy(sim_job, policy_uri, cfg, logger)
        all_results["policies"].append(policy_results)

    # Always output JSON results to stdout
    # Ensure all logging is flushed before printing JSON
    import sys

    sys.stderr.flush()
    sys.stdout.flush()

    # Print JSON with a marker for easier extraction
    print("===JSON_OUTPUT_START===")
    print(json.dumps(all_results, indent=2))
    print("===JSON_OUTPUT_END===")


if __name__ == "__main__":
    main()
