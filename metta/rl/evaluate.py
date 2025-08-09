"""Policy evaluation functionality."""

import logging
import uuid

import numpy as np
import torch
import wandb

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskResponse
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy as eval_service_evaluate_policy
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.utils import get_or_create_policy_ids, wandb_policy_name_to_uri

logger = logging.getLogger(__name__)


def evaluate_policy_remote(
    policy_record: PolicyRecord,
    sim_suite_config: SimulationSuiteConfig,
    stats_epoch_id: uuid.UUID | None,
    wandb_policy_name: str | None,
    stats_client: StatsClient | None,
    wandb_run: WandbRun | None,
    trainer_cfg: TrainerConfig,
) -> TaskResponse | None:
    """Create a task to evaluate a policy remotely.

    Ensures policy is uploaded to wandb.

    Returns:
        TaskResponse for the policy evaluation or None if policy is not uploaded to wandb
    """
    if wandb_run and stats_client and policy_record and wandb_policy_name:
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
                        sim_suite=sim_suite_config.name,
                        attributes={
                            "sim_suite_config": sim_suite_config.model_dump(mode="json"),
                            "git_hash": trainer_cfg.simulation.git_hash,
                        },
                    )
                )
                logger.info(
                    f"Policy evaluator: created task {task.id} for {wandb_policy_name} on {sim_suite_config.name}"
                )
                # TODO: create a task for the trainer curriculum. Sample a few. Also pass
                # along overrides if needed. Should make sure we handle overrides correctly
                # in eval_task_worker.py's call out to sim.py (and make sure sim.py handles)
                # them correctly. Also these tasks should get registered as eval/training_task

                return task
        return None


def evaluate_policy(
    *,
    policy_record: PolicyRecord,
    sim_suite_config: SimulationSuiteConfig,
    device: torch.device,
    vectorization: str,
    replay_dir: str | None,
    stats_epoch_id: uuid.UUID | None,
    wandb_policy_name: str | None,
    policy_store: PolicyStore,
    stats_client: StatsClient | None,
    wandb_run: WandbRun | None,
    trainer_cfg: TrainerConfig,
    agent_step: int,
    epoch: int,
) -> EvalRewardSummary:
    """Evaluate policy using the eval service, handling scoring and replay uploads.

    This function orchestrates policy evaluation including:
    - Remote evaluation via stats server if configured
    - Local evaluation using the eval service
    - Policy metadata scoring for sweep evaluations
    - Replay HTML upload to wandb

    Returns:
        EvalRewardSummary containing the evaluation scores
    """
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

    category_scores = list(eval_scores.category_scores.values())
    if category_scores and policy_record:
        policy_record.metadata["score"] = float(np.mean(category_scores))

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
    wandb_run: WandbRun,
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
