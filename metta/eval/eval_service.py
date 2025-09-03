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

    # Load the policy from URI directly to the correct device
    policy = CheckpointManager.load_from_uri(checkpoint_uri, device=device)

    sims = [
        Simulation(
            name=sim.name,
            cfg=sim,
            policy=policy,
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
            logger.info("=== Simulation '%s' ===", sim.name)
            sim_result = sim.simulate()
            merged_db.merge_in(sim_result.stats_db)
            record_heartbeat()
            if replay_dir is not None:
                checkpoint_metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)
                key, version = checkpoint_metadata["run_name"], checkpoint_metadata["epoch"]
                sim_replay_urls = sim_result.stats_db.get_replay_urls(key, version)
                if sim_replay_urls:
                    replay_urls[sim.name] = sim_replay_urls
                    logger.info(f"Collected {len(sim_replay_urls)} replay URL(s) for simulation '{sim.name}'")
            sim_result.stats_db.close()
            successful_simulations += 1
        except SimulationCompatibilityError as e:
            # Only skip for NPC-related compatibility issues
            error_msg = str(e).lower()
            if "npc" in error_msg or "non-player" in error_msg:
                logger.warning("Skipping simulation '%s' due to NPC compatibility issue: %s", sim.name, str(e))
                continue
            else:
                # Re-raise for non-NPC compatibility issues
                logger.error("Critical compatibility error in simulation '%s': %s", sim.name, str(e))
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
    categories: set[str] = set()
    for sim_config in simulations:
        categories.add(sim_config.name.split("/")[0])

    category_scores: dict[str, float] = {}
    for category in categories:
        score = stats_db.get_average_metric("reward", checkpoint_uri, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        if score is None:
            continue
        category_scores[category] = score
    per_sim_scores: dict[tuple[str, str], float] = {}
    all_scores = stats_db.simulation_scores(checkpoint_uri, "reward")
    for (sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        per_sim_scores[(category, sim_short_name)] = score

    return EvalRewardSummary(
        category_scores=category_scores,
        simulation_scores=per_sim_scores,
    )
