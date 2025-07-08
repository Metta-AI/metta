import uuid

import torch
from pydantic import ConfigDict, Field

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.script_decorators import get_metta_logger
from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.simulation_suite import SimulationSuite


class EvaluationJob(Config):
    __init__ = Config.__init__

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    policy_record: PolicyRecord
    simulation_suite: SimulationSuiteConfig
    stats_dir: str
    replay_dir: str
    device: str
    vectorization: str
    data_dir: str
    export_stats_db_uri: str | None = None
    stats_epoch_id: uuid.UUID | None = None
    wandb_policy_name: str | None = None
    extract_replay_url: bool = True


class EvaluationScores(BaseModelWithForbidExtra):
    suite_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each sim suite")
    simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward for each sim environment (keyed on (suite_name, sim_name))"
    )


class EvaluationResults(BaseModelWithForbidExtra):
    scores: EvaluationScores = Field(..., description="Evaluation scores")
    replay_url: str | None = Field(..., description="Replay URL")


def evaluate_policy(
    job: EvaluationJob,
    policy_store: PolicyStore,
    stats_client: StatsClient | None,
) -> EvaluationResults:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.

    Returns:
        Dictionary containing simulation results and metrics
    """
    logger = get_metta_logger()
    pr = job.policy_record

    # For each checkpoint of the policy, simulate
    logger.info(f"Evaluating policy {pr.uri}")
    replay_dir = f"{job.replay_dir}/{pr.run_name}"
    sim = SimulationSuite(
        config=job.simulation_suite,
        policy_pr=pr,
        policy_store=policy_store,
        replay_dir=replay_dir,
        stats_dir=job.stats_dir,
        device=torch.device(job.device),
        vectorization=job.vectorization,
        stats_client=stats_client,
        stats_epoch_id=job.stats_epoch_id,
        wandb_policy_name=job.wandb_policy_name,
    )
    sim_results = sim.simulate()

    eval_stats_db = EvalStatsDB.from_sim_stats_db(sim_results.stats_db)
    scores = extract_scores(job, eval_stats_db)
    logger.info("Evaluation complete for policy %s", pr.uri)

    if job.export_stats_db_uri is not None:
        logger.info("Exporting merged stats DB → %s", job.export_stats_db_uri)
        sim_results.stats_db.export(job.export_stats_db_uri)

    if job.extract_replay_url:
        logger.info("Generating replay URL")
        replay_url = extract_replay_url(sim_results.stats_db, pr)
    else:
        replay_url = None

    results = EvaluationResults(
        scores=scores,
        replay_url=replay_url,
    )

    return results


def extract_replay_url(stats_db: SimulationStatsDB, policy_pr: PolicyRecord) -> str | None:
    key, version = stats_db.key_and_version(policy_pr)
    replay_urls = stats_db.get_replay_urls(key, version)
    if len(replay_urls) > 0:
        return replay_urls[0]
    return None


def extract_scores(job: EvaluationJob, stats_db: EvalStatsDB) -> EvaluationScores:
    logger = get_metta_logger()
    categories: set[str] = set()
    for sim_name in job.simulation_suite.simulations.keys():
        categories.add(sim_name.split("/")[0])

    category_scores: dict[str, float] = {}
    for category in categories:
        score = stats_db.get_average_metric_by_filter("reward", job.policy_record, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        if score is None:
            continue
        category_scores[category] = score
    per_sim_scores: dict[tuple[str, str], float] = {}
    all_scores = stats_db.simulation_scores(job.policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        per_sim_scores[(category, sim_short_name)] = score

    return EvaluationScores(
        suite_scores=category_scores,
        simulation_scores=per_sim_scores,
    )
