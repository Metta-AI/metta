"""Policy evaluation functionality."""

import logging
import uuid
from typing import Any

import wandb

from metta.agent.policy_record import PolicyRecord
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskResponse
from metta.common.util.collections import remove_none_keys
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.common.wandb.wandb_context import WandbRun
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
                            "sim_suite_config": sim_suite_config.to_jsonable(),
                            "git_hash": trainer_cfg.simulation.git_hash,
                            "trainer_task": {
                                "curriculum": trainer_cfg.curriculum_or_env,
                                "env_overrides": trainer_cfg.env_overrides,
                            },
                        },
                    )
                )
                logger.info(
                    f"Policy evaluator: created task {task.id} for {wandb_policy_name} on {sim_suite_config.name}"
                )

                return task
        return None


def upload_replay_html(
    replay_urls: dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: WandbRun,
    metric_prefix: str | None = None,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
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
                links.append(_form_mettascope_link(urls[0], name))
            else:
                # Multiple episodes - show name with numbered links
                episode_links = []
                for i, url in enumerate(urls, 1):
                    episode_links.append(_form_mettascope_link(url, str(i)))
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Log all links in a single HTML entry
        html_content = " | ".join(links)
        _upload_replay_html(
            html_content, agent_step, epoch, wandb_run, metric_prefix, step_metric_key, epoch_metric_key
        )

    # Maintain backward compatibility - log training task separately if available
    if "eval/training_task" in replay_urls and replay_urls["eval/training_task"]:
        training_url = replay_urls["eval/training_task"][0]  # Use first URL for backward compatibility
        html_content = _form_mettascope_link(training_url, f"MetaScope Replay (Epoch {epoch})")
        _upload_replay_html(
            html_content, agent_step, epoch, wandb_run, metric_prefix, step_metric_key, epoch_metric_key
        )


def _form_mettascope_link(url: str, name: str) -> str:
    return f'<a href="{METTASCOPE_REPLAY_URL}/?replayUrl={url}" target="_blank">{name}</a>'


def _upload_replay_html(
    html_content: str,
    agent_step: int,
    epoch: int,
    wandb_run: WandbRun,
    metric_prefix: str | None = None,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
) -> None:
    key_all = (f"{metric_prefix}/" if metric_prefix else "") + "replays/all"
    payload: dict[str, Any] = remove_none_keys(
        {key_all: wandb.Html(html_content), step_metric_key: agent_step, epoch_metric_key: epoch}
    )
    if step_metric_key or epoch_metric_key:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=agent_step)
