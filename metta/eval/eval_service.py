import logging
import uuid
from pathlib import Path

import torch

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.collections import is_unique
from metta.common.util.heartbeat import record_heartbeat
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation import Simulation, SimulationCompatibilityError
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB

logger = logging.getLogger(__name__)


def evaluate_policy(
    *,
    checkpoint_uri: str,
    simulations: list[SimulationConfig],
    device: torch.device,
    vectorization: str,
    stats_dir: str | None = None,
    replay_dir: str | None = None,
    export_stats_db_uri: str | None = None,
    stats_epoch_id: uuid.UUID | None = None,
    eval_task_id: uuid.UUID | None = None,
    stats_client: StatsClient | None,
) -> EvalResults:
    """Evaluate one policy URI, merging all simulations into a single StatsDB."""
    stats_dir = stats_dir or "/tmp/stats"

    logger.info(f"Evaluating checkpoint {checkpoint_uri}")
    if not is_unique([sim.name for sim in simulations]):
        raise ValueError("Simulation names must be unique")

    sims = [
        Simulation(
            cfg=sim,
            policy_uri=checkpoint_uri,
            device=device,
            vectorization=vectorization,
            stats_dir=stats_dir,
            replay_dir=replay_dir,
            stats_client=stats_client,
            stats_epoch_id=stats_epoch_id,
            eval_task_id=eval_task_id,
        )
        for sim in simulations
    ]
    successful_simulations = 0
    replay_urls: dict[str, list[str]] = {}
    merged_db: SimulationStatsDB = SimulationStatsDB(Path(f"{stats_dir}/all_{uuid.uuid4().hex[:8]}.duckdb"))
    for sim in sims:
        try:
            record_heartbeat()
            logger.info("=== Simulation '%s' ===", sim.full_name)
            sim_result = sim.simulate()
            record_heartbeat()
            if replay_dir is not None:
                sim_replay_urls = sim_result.stats_db.get_replay_urls(
                    policy_uri=checkpoint_uri, sim_suite=sim._config.suite, env=sim._config.name
                )
                if sim_replay_urls:
                    replay_urls[sim.full_name] = sim_replay_urls
                    logger.info(f"Collected {len(sim_replay_urls)} replay URL(s) for simulation '{sim.full_name}'")
            sim_result.stats_db.close()
            merged_db.merge_in(sim_result.stats_db)
            successful_simulations += 1
        except SimulationCompatibilityError as e:
            # Only skip for NPC-related compatibility issues
            error_msg = str(e).lower()
            if "npc" in error_msg or "non-player" in error_msg:
                logger.warning("Skipping simulation '%s' due to NPC compatibility issue: %s", sim.full_name, str(e))
                continue
            else:
                # Re-raise for non-NPC compatibility issues
                logger.error("Critical compatibility error in simulation '%s': %s", sim.full_name, str(e))
                raise
    if successful_simulations == 0:
        raise RuntimeError("No simulations could be run successfully")
    logger.info("Completed %d/%d simulations successfully", successful_simulations, len(simulations))

    eval_stats_db = EvalStatsDB(merged_db.path)
    logger.info("Evaluation complete for checkpoint %s", checkpoint_uri)
    scores = extract_scores(checkpoint_uri, simulations, eval_stats_db)

    if export_stats_db_uri is not None:
        logger.info("Exporting merged stats DB → %s", export_stats_db_uri)
        merged_db.export(export_stats_db_uri)

    results = EvalResults(
        scores=scores,
        replay_urls=replay_urls,
    )

    return results


def extract_scores(
    checkpoint_uri: str, simulations: list[SimulationConfig], stats_db: EvalStatsDB
) -> EvalRewardSummary:
    suites = {sim_config.suite for sim_config in simulations}

    def suite_score(suite: str) -> float | None:
        score = stats_db.get_average_metric("reward", checkpoint_uri, f"sim_name LIKE '%{suite}%'")
        logger.info(f"{suite} score: {score}")
        return score

    category_scores = {suite: score for suite in suites if (score := suite_score(suite)) is not None}

    return EvalRewardSummary(
        category_scores=category_scores,
        simulation_scores=stats_db.simulation_scores(checkpoint_uri, "reward"),
    )
