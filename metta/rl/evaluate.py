import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import wandb

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskResponse
from metta.common.util.collections import remove_none_keys
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.common.wandb.wandb_context import WandbRun
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sim.utils import get_or_create_policy_ids

logger = logging.getLogger(__name__)


def evaluate_policy_remote_with_checkpoint_manager(
    checkpoint_manager: CheckpointManager,
    checkpoint_path: Optional[str],
    simulations: list[SimulationConfig],
    stats_epoch_id: uuid.UUID | None,
    policy_uri: str | None,
    stats_client: StatsClient | None,
    wandb_run: WandbRun | None,
    trainer_cfg: TrainerConfig,
) -> TaskResponse | None:
    """Create a remote evaluation task using CheckpointManager.

    Args:
        checkpoint_manager: CheckpointManager instance
        checkpoint_path: Specific checkpoint path, or None for latest
        simulations: List of simulations to run
        stats_epoch_id: Stats epoch ID for tracking
        policy_uri: Policy URI (wandb://, file://, etc.)
        stats_client: Client for stats server communication
        wandb_run: WandB run context
        trainer_cfg: Training configuration

    Returns:
        TaskResponse if evaluation task created, None otherwise
    """
    if not (wandb_run and stats_client and policy_uri):
        logger.warning("Remote evaluation requires wandb_run, stats_client, and policy_uri")
        return None

    # Get checkpoint path if not specified
    if checkpoint_path is None:
        checkpoint_path = checkpoint_manager.find_best_checkpoint("score")
        if checkpoint_path is None:
            logger.warning("No checkpoints available for remote evaluation")
            return None

    # Normalize the policy URI
    normalized_uri = CheckpointManager.normalize_uri(policy_uri)

    # For wandb URIs, ensure they have a version
    if normalized_uri.startswith("wandb://") and ":" not in normalized_uri:
        logger.warning(f"Remote evaluation: {normalized_uri} does not specify a version")
        return None

    # Process policy registration using the new format
    stats_server_policy_id = get_or_create_policy_ids(
        stats_client,
        [(normalized_uri, wandb_run.notes)],  # New format: (uri, description)
        stats_epoch_id,
    ).get(CheckpointManager.get_policy_metadata(normalized_uri)["run_name"])

    if not stats_server_policy_id:
        logger.warning(f"Remote evaluation: failed to get or register policy ID for {normalized_uri}")
        return None

    # Create evaluation task
    task = stats_client.create_task(
        TaskCreateRequest(
            policy_id=stats_server_policy_id,
            sim_suite=simulations[0].name,
            attributes={
                "git_hash": (trainer_cfg.evaluation and trainer_cfg.evaluation.git_hash),
                "simulations": [sim.model_dump() for sim in simulations],
            },
        )
    )

    logger.info(
        f"Policy evaluator: created task {task.id} for {normalized_uri} on {simulations[0].name} "
        f"using checkpoint {Path(checkpoint_path).name}"
    )

    return task


def upload_replay_html(
    replay_urls: dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: WandbRun,
    metric_prefix: str | None = None,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
) -> None:
    """Upload organized replay HTML links to wandb."""
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
                links.append(_form_mettascope_link(urls[0], name))
            else:
                # Multiple episodes - show name with numbered links
                episode_links = []
                for i, url in enumerate(urls, 1):
                    episode_links.append(_form_mettascope_link(url, str(i)))
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Log all links in a single HTML entry
        html_content = " | ".join(links)
        _upload_replay_html(html_content, agent_step, epoch, wandb_run, step_metric_key, epoch_metric_key)


def _form_mettascope_link(url: str, name: str) -> str:
    return f'<a href="{METTASCOPE_REPLAY_URL}/?replayUrl={url}" target="_blank">{name}</a>'


def _upload_replay_html(
    html_content: str,
    agent_step: int,
    epoch: int,
    wandb_run: WandbRun,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
) -> None:
    payload: dict[str, Any] = remove_none_keys(
        {"replays/all": wandb.Html(html_content), step_metric_key: agent_step, epoch_metric_key: epoch}
    )
    if step_metric_key or epoch_metric_key:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=epoch)
