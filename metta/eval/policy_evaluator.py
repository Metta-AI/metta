from __future__ import annotations

import logging
import uuid
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOn, WandbContext
from metta.eval.evaluation_service import EvaluationResults, EvaluationService
from metta.sim.simulation_config import SimulationSuiteConfig


class EvaluateJobConfig(Config):
    __init__ = Config.__init__

    # This is temporary. Helps keep evaluate_job.yaml simple for now.
    model_config = {"extra": "ignore"}

    policy_uri: str
    sim: SimulationSuiteConfig
    stats_dir: str
    stats_db_path: str
    replay_dir: str
    upload_to_wandb: bool
    wandb: WandbConfig
    device: str
    vectorization: str
    data_dir: str
    stats_server_uri: str | None = None

    @property
    def wandb_policy_name(self) -> str | None:
        """
        - wandb://project/artifact:version -> entity/project/artifact:version
        - wandb://entity/project/artifact:version -> entity/project/artifact:version
        """
        wandb_prefix = "wandb://"
        if not self.policy_uri.startswith(wandb_prefix):
            return None

        uri_body = self.policy_uri[len(wandb_prefix) :]
        parts = uri_body.split("/")

        if len(parts) == 2:
            # Format: project/artifact:version
            # Need entity from config
            entity = getattr(self.wandb, "entity", None)
            if entity:
                return f"{entity}/{uri_body}"
        elif len(parts) >= 3:
            # Format: entity/project/artifact:version or entity/project/artifact_type/name:version
            # Entity is already in the URI
            return uri_body

        return None


def evaluate_policy(config: EvaluateJobConfig, logger: logging.Logger) -> EvaluationResults:
    """
    Evaluate a single policy and return results.
    """
    logger.info(f"Evaluating policy: {config.policy_uri}")

    # Create minimal DictConfig for PolicyStore compatibility
    minimal_cfg = DictConfig(
        {
            "device": config.device,
            "data_dir": config.data_dir,
            "wandb": config.wandb.model_dump(),
            "stats_server_uri": config.stats_server_uri,
        }
    )

    policy_store = PolicyStore(minimal_cfg, None)
    policy_pr = policy_store.policy_record(config.policy_uri)
    logger.info(f"Loaded policy: {policy_pr.run_name}")

    Path(config.stats_dir).mkdir(parents=True, exist_ok=True)
    stats_client = get_stats_client(minimal_cfg, logger)

    eval_service = EvaluationService(
        policy_store=policy_store,
        device=config.device,
        vectorization=config.vectorization,
        stats_client=stats_client,
        logger=logger,
    )
    results: EvaluationResults = eval_service.run_evaluation(
        policy_pr=policy_pr,
        sim_config=config.sim,
        stats_dir=config.stats_dir,
        replay_dir=config.replay_dir,
        stats_db_path=config.stats_db_path,
        wandb_policy_name=config.wandb_policy_name,
    )

    logger.info("Evaluation scores:")
    for score_name, score_value in sorted(results.scores.simulation_scores.items()):
        logger.info(f"  {score_name}: {score_value:.4f}")

    if config.upload_to_wandb:
        upload_results_to_wandb(results, config, logger)

    return results


def upload_results_to_wandb(results: EvaluationResults, config: EvaluateJobConfig, logger: logging.Logger):
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
                "policy_uri": config.policy_uri,
                "sim_suite": OmegaConf.to_container(config.sim, resolve=False),
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
                "policy_uri": config.policy_uri,
                "scores": results.scores.model_dump(),
            },
        )
        artifact.add_file(config.stats_db_path)
        wandb.log_artifact(artifact)

        if wandb.run:
            logger.info(f"Results uploaded to wandb: {wandb.run.url}")
