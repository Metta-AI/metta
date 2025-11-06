import logging
import pathlib
import uuid

import metta.app_backend.clients.stats_client
import metta.common.util.collections
import metta.common.util.heartbeat
import metta.eval.eval_request_config
import metta.eval.eval_stats_db
import metta.sim.simulation
import metta.sim.simulation_config
import metta.sim.simulation_stats_db

logger = logging.getLogger(__name__)


def evaluate_policy(
    *,
    checkpoint_uri: str,
    simulations: list[metta.sim.simulation_config.SimulationConfig],
    replay_dir: str | None,
    stats_dir: str = "/tmp/stats",
    export_stats_db_uri: str | None = None,
    stats_epoch_id: uuid.UUID | None = None,
    eval_task_id: uuid.UUID | None = None,
    stats_client: metta.app_backend.clients.stats_client.StatsClient | None,
) -> metta.eval.eval_request_config.EvalResults:
    """Evaluate one policy URI, merging all simulations into a single StatsDB."""
    logger.info(f"Evaluating checkpoint {checkpoint_uri}")
    if not metta.common.util.collections.is_unique([sim.full_name for sim in simulations]):
        raise ValueError("Simulation names must be unique")

    sims = [
        metta.sim.simulation.Simulation(
            cfg=sim,
            policy_uri=checkpoint_uri,
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
    merged_db: metta.sim.simulation_stats_db.SimulationStatsDB = metta.sim.simulation_stats_db.SimulationStatsDB(
        pathlib.Path(f"{stats_dir}/all_{uuid.uuid4().hex[:8]}.duckdb")
    )
    for sim in sims:
        try:
            metta.common.util.heartbeat.record_heartbeat()
            logger.info("=== Simulation '%s' ===", sim.full_name)
            sim_result = sim.simulate()
            metta.common.util.heartbeat.record_heartbeat()
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
        except metta.sim.simulation.SimulationCompatibilityError as e:
            # Only skip for NPC-related compatibility issues
            error_msg = str(e).lower()
            if "npc" in error_msg or "non-player" in error_msg:
                logger.warning("Skipping simulation '%s' due to NPC compatibility issue: %s", sim.full_name, str(e))
                continue
            else:
                # Re-raise for non-NPC compatibility issues
                logger.error(
                    "Critical compatibility error in simulation '%s': %s", sim.full_name, str(e), exc_info=True
                )
                raise
    if successful_simulations == 0:
        raise RuntimeError("No simulations could be run successfully")
    logger.info("Completed %d/%d simulations successfully", successful_simulations, len(simulations))

    eval_stats_db = metta.eval.eval_stats_db.EvalStatsDB(merged_db.path)
    logger.info("Evaluation complete for checkpoint %s", checkpoint_uri)
    scores = extract_scores(checkpoint_uri, simulations, eval_stats_db)

    if export_stats_db_uri is not None:
        logger.info("Exporting merged stats DB → %s", export_stats_db_uri)
        merged_db.export(export_stats_db_uri)

    results = metta.eval.eval_request_config.EvalResults(
        scores=scores,
        replay_urls=replay_urls,
    )

    return results


def extract_scores(
    checkpoint_uri: str,
    simulations: list[metta.sim.simulation_config.SimulationConfig],
    stats_db: metta.eval.eval_stats_db.EvalStatsDB,
) -> metta.eval.eval_request_config.EvalRewardSummary:
    suites = {sim_config.suite for sim_config in simulations}

    def suite_score(suite: str) -> float | None:
        score = stats_db.get_average_metric("reward", checkpoint_uri, f"sim_name LIKE '%{suite}%'")
        logger.info(f"{suite} score: {score}")
        return score

    category_scores = {suite: score for suite in suites if (score := suite_score(suite)) is not None}

    return metta.eval.eval_request_config.EvalRewardSummary(
        category_scores=category_scores,
        simulation_scores=stats_db.simulation_scores(checkpoint_uri, "reward"),
    )
