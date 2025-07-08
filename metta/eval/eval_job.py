import uuid

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, field_serializer

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.script_decorators import get_metta_logger
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOn, WandbContext
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.simulation_suite import SimulationSuite


class EvaluationJob(Config):
    __init__ = Config.__init__

    policy_checkpoint_uri: str  # Corresponds to a single policy checkpoint

    simulation_suite: SimulationSuiteConfig
    stats_dir: str
    stats_db_uri: str
    replay_dir: str
    upload_to_wandb: bool
    wandb: WandbConfig
    device: str
    vectorization: str
    data_dir: str
    stats_server_uri: str | None = None


class EvaluationScores(BaseModelWithForbidExtra):
    overall_score: float | None = Field(default=None, description="Overall average score across all suites")
    suite_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each sim suite")
    simulation_scores: dict[str, float] = Field(
        default_factory=dict, description="Average reward for each sim environment (key format: suite_name/sim_name)"
    )


class EvaluationResults(BaseModelWithForbidExtra):
    policy_record: PolicyRecord = Field(..., description="Policy record")
    scores: EvaluationScores = Field(..., description="Evaluation scores")
    replay_url: str | None = Field(..., description="Replay URL")

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @field_serializer("policy_record")
    def serialize_policy_record(self, value: PolicyRecord):
        return {
            "name": value.run_name,
            "uri": value.uri,
        }


def evaluate_policy(
    config: EvaluationJob,
) -> EvaluationResults:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.

    Returns:
        Dictionary containing simulation results and metrics
    """
    logger = get_metta_logger()
    minimal_cfg = DictConfig(
        {
            "device": config.device,
            "data_dir": config.data_dir,
            "wandb": config.wandb.model_dump(),
            "stats_server_uri": config.stats_server_uri,
        }
    )
    policy_store = PolicyStore(minimal_cfg, None)
    stats_client: StatsClient | None = get_stats_client(minimal_cfg, logger)
    pr = policy_store.policy_record(config.policy_checkpoint_uri)

    # For each checkpoint of the policy, simulate
    logger.info(f"Evaluating policy {pr.uri}")
    replay_dir = f"{config.replay_dir}/{pr.run_name}"
    sim = SimulationSuite(
        config=config.simulation_suite,
        policy_pr=pr,
        policy_store=policy_store,
        replay_dir=replay_dir,
        stats_dir=config.stats_dir,
        device=torch.device(config.device),
        vectorization=config.vectorization,
        stats_client=stats_client,
    )
    sim_results = sim.simulate()

    eval_stats_db = EvalStatsDB.from_sim_stats_db(sim_results.stats_db)
    scores = extract_scores(eval_stats_db, pr)
    logger.info("Exporting merged stats DB → %s", config.stats_db_uri)
    sim_results.stats_db.export(config.stats_db_uri)
    logger.info("Evaluation complete for policy %s", pr.uri)
    replay_url = extract_replay_url(sim_results.stats_db, pr)
    results = EvaluationResults(
        policy_record=pr,
        scores=scores,
        replay_url=replay_url,
    )

    if config.upload_to_wandb:
        upload_results_to_wandb(results, config)

    return results


def extract_replay_url(stats_db: SimulationStatsDB, policy_pr: PolicyRecord) -> str | None:
    key, version = stats_db.key_and_version(policy_pr)
    replay_urls = stats_db.get_replay_urls(key, version)
    if len(replay_urls) > 0:
        return replay_urls[0]
    return None


def extract_scores(stats_db: EvalStatsDB, policy_pr: PolicyRecord) -> EvaluationScores:
    logger = get_metta_logger()
    if policy_pr.uri is None:
        logger.warning("Policy URI is None, cannot extract scores")
        return EvaluationScores()

    suite_scores = {}
    simulation_scores = {}
    overall_score = None

    # Get all locally run simulations for this policy
    sims_df = stats_db.query(f"""
        SELECT DISTINCT sim_suite, sim_name
        FROM policy_simulation_agent_samples
        WHERE policy_key = '{policy_pr.uri}' AND policy_version = {policy_pr.metadata.epoch}
    """)

    if len(sims_df) == 0:
        logger.warning("No simulations found in database")
        return EvaluationScores()

    # Group by simulation suite (category)
    suites = sims_df["sim_suite"].unique()
    for suite in suites:
        suite_score = stats_db.get_average_metric_by_filter("reward", policy_pr, f"sim_suite = '{suite}'")

        if suite_score is not None:
            suite_scores[suite] = suite_score
            # Get individual simulation scores within this suite
            suite_sims = sims_df[sims_df["sim_suite"] == suite]["sim_name"].unique()

            for sim_name in suite_sims:
                sim_score = stats_db.get_average_metric_by_filter(
                    "reward", policy_pr, f"sim_suite = '{suite}' AND sim_name = '{sim_name}'"
                )
                if sim_score is not None:
                    simulation_scores[f"{suite}/{sim_name}"] = sim_score

    # Calculate overall score as average of suite scores (matching trainer)
    suite_score_values = list(suite_scores.values())
    if suite_score_values:
        overall_score = sum(suite_score_values) / len(suite_score_values)

    return EvaluationScores(
        overall_score=overall_score,
        suite_scores=suite_scores,
        simulation_scores=simulation_scores,
    )


def upload_results_to_wandb(results: EvaluationResults, config: EvaluationJob):
    logger = get_metta_logger()
    if not config.upload_to_wandb:
        logger.info("Wandb upload is disabled, skipping upload")
        return
    elif config.wandb.enabled:
        assert isinstance(config.wandb, WandbConfigOn), "Wandb config must be enabled"
    else:
        logger.info("Wandb is disabled, skipping upload")
        return

    wandb_config = {
        "mode": "online",
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "name": f"eval_{results.policy_record.run_name}_{uuid.uuid4().hex[:8]}",
        "tags": ["evaluation", "policy_evaluator"],
    }

    with WandbContext(DictConfig(wandb_config), None):
        wandb.config.update(
            {
                "policy_uri": results.policy_record.uri,
                "sim_suite": OmegaConf.to_container(config.simulation_suite, resolve=False),
                "device": config.device,
            }
        )

        metrics = {}
        for suite_name, suite_score in results.scores.suite_scores.items():
            metrics[f"eval_{suite_name}/score"] = suite_score

        for sim_full_name, sim_score in results.scores.simulation_scores.items():
            # sim_full_name is like "navigation/maze_easy"
            metrics[f"eval_{sim_full_name}"] = sim_score

        if metrics:
            wandb.log(metrics)
            logger.info(f"Logged {len(metrics)} evaluation metrics to wandb")

        # Extract and upload replay URL if available
        if results.replay_url:
            metascope_url = f"https://metta-ai.github.io/metta/?replayUrl={results.replay_url}"
            # Log as HTML link (matching trainer format)
            wandb.log({"replays/link": wandb.Html(f'<a href="{metascope_url}">MetaScope Replay (Evaluation)</a>')})
            logger.info(f"Uploaded replay link to wandb: {metascope_url}")

        # Upload stats database as artifact (keep this extra feature)
        artifact = wandb.Artifact(
            name=f"eval_stats_{results.policy_record.run_name}",
            type="evaluation_stats",
            metadata={
                "policy_uri": results.policy_record.uri,
                "scores": results.scores.model_dump(),
            },
        )
        artifact.add_file(config.stats_db_uri)
        wandb.log_artifact(artifact)

        if wandb.run:
            logger.info(f"Results uploaded to wandb: {wandb.run.url}")
