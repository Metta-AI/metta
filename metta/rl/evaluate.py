"""Policy evaluation functionality."""

import logging
import uuid
from typing import Any

import numpy as np
import torch
import wandb

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy as eval_service_evaluate_policy
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.utils import get_or_create_policy_ids, wandb_policy_name_to_uri

logger = logging.getLogger(__name__)


def evaluate_policy(
    *,
    policy_record: Any,
    policy_uri: str,
    sim_suite_config: SimulationSuiteConfig,
    device: torch.device,
    vectorization: str,
    replay_dir: str | None,
    stats_epoch_id: uuid.UUID | None,
    wandb_policy_name: str | None,
    policy_store: Any,
    stats_client: StatsClient | None,
    cfg: Any,
    wandb_run: Any | None,
    trainer_cfg: Any,
    agent_step: int,
    epoch: int,
) -> EvalRewardSummary:
    """Evaluate policy using the eval service and handle remote evaluation, scoring, and replay uploads.

    This function orchestrates policy evaluation including:
    - Remote evaluation via stats server if configured
    - Local evaluation using the eval service
    - Policy metadata scoring for sweep evaluations
    - Replay HTML upload to wandb

    Returns:
        EvalRewardSummary containing the evaluation scores
    """
    # Handle remote evaluation if configured
    if (
        trainer_cfg.simulation.evaluate_remote
        and wandb_run
        and stats_client
        and policy_record
        and wandb_policy_name  # ensures it was uploaded to wandb
    ):
        # Need to upload policy artifact to wandb first and make sure our name
        # reflects that in the version
        if ":" not in wandb_policy_name:
            logger.warning(f"Remote evaluation: {wandb_policy_name} does not specify a version")
        else:
            internal_wandb_policy_name, wandb_uri = wandb_policy_name_to_uri(wandb_policy_name)
            stats_server_policy_id = get_or_create_policy_ids(
                stats_client,
                [(internal_wandb_policy_name, wandb_uri, wandb_run.notes)],
                stats_epoch_id,
            ).get(internal_wandb_policy_name)
            if not stats_server_policy_id:
                logger.warning(f"Remote evaluation: failed to get or register policy ID for {wandb_policy_name}")
            else:
                task = stats_client.create_task(
                    TaskCreateRequest(
                        policy_id=stats_server_policy_id,
                        git_hash=trainer_cfg.simulation.git_hash,
                        sim_suite=sim_suite_config.name,
                    )
                )
                logger.info(f"Remote evaluation: created task {task.id} for policy {wandb_policy_name}")
                # TODO: need policy evaluator to generate replays and push stats to wandb

    # Local evaluation
    logger.info(f"Simulating policy: {policy_uri} with extended config including training task")
    evaluation_results = eval_service_evaluate_policy(
        policy_record=policy_record,
        simulation_suite=sim_suite_config,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,  # Pass replay_dir to enable replay generation
        stats_epoch_id=stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
        policy_store=policy_store,
        stats_client=stats_client,
        logger=logger,
    )
    logger.info("Simulation complete")

    eval_scores = evaluation_results.scores

    # Get target metric (for logging) from sweep config
    # and write top-level score for policy selection.
    # In sweep_eval, we use the "score" entry in the policy metadata to select the best policy
    target_metric = getattr(cfg, "sweep", {}).get("metric", "reward")  # fallback to reward
    category_scores = list(eval_scores.category_scores.values())
    if category_scores and policy_record:
        policy_record.metadata["score"] = float(np.mean(category_scores))
        logger.info(f"Set policy metadata score to {policy_record.metadata['score']} using {target_metric} metric")

    # Generate and upload replay HTML if we have wandb
    if wandb_run is not None and evaluation_results.replay_urls:
        upload_replay_html(
            replay_urls=evaluation_results.replay_urls,
            agent_step=agent_step,
            epoch=epoch,
            wandb_run=wandb_run,
        )

    return eval_scores


def upload_replay_html(
    replay_urls: dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: Any,
) -> None:
    """Upload replay HTML to wandb with organized links."""
    # Create unified HTML with all replay links on a single line
    if replay_urls:
        # Group replays by base name
        replay_groups = {}

        for sim_name, urls in sorted(replay_urls.items()):
            if "training_task" in sim_name:
                # Training replays
                if "training" not in replay_groups:
                    replay_groups["training"] = []
                replay_groups["training"].extend(urls)
            else:
                # Evaluation replays - clean up the display name
                display_name = sim_name.replace("eval/", "")
                if display_name not in replay_groups:
                    replay_groups[display_name] = []
                replay_groups[display_name].extend(urls)

        # Build HTML with episode numbers
        links = []
        for name, urls in replay_groups.items():
            if len(urls) == 1:
                # Single episode - just show the name
                player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={urls[0]}"
                links.append(f'<a href="{player_url}" target="_blank">{name}</a>')
            else:
                # Multiple episodes - show name with numbered links
                episode_links = []
                for i, url in enumerate(urls, 1):
                    player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={url}"
                    episode_links.append(f'<a href="{player_url}" target="_blank">{i}</a>')
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Log all links in a single HTML entry
        html_content = " | ".join(links)
        wandb_run.log(
            {"replays/all": wandb.Html(html_content)},
            step=agent_step,
        )

    # Maintain backward compatibility - log training task separately if available
    if "eval/training_task" in replay_urls and replay_urls["eval/training_task"]:
        training_url = replay_urls["eval/training_task"][0]  # Use first URL for backward compatibility
        player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={training_url}"
        link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
        wandb_run.log(link_summary, step=agent_step)
