"""Policy evaluation functionality."""

import logging
from typing import Any

import wandb

from metta.common.util.collections import remove_none_keys
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.common.wandb.wandb_context import WandbRun

logger = logging.getLogger(__name__)


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

    # Maintain backward compatibility - log training task separately if available
    if "eval/training_task" in replay_urls and replay_urls["eval/training_task"]:
        training_url = replay_urls["eval/training_task"][0]  # Use first URL for backward compatibility
        html_content = _form_mettascope_link(training_url, f"MetaScope Replay (Epoch {epoch})")
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
