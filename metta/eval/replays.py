"""Replay generation and management functionality for Metta."""

import logging
from typing import Any

import wandb

from metta.common.util.constants import METTASCOPE_REPLAY_URL

logger = logging.getLogger(__name__)


def upload_replay_html(
    replay_urls: dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: Any,
) -> None:
    """Upload replay HTML to wandb with unified view of all replay links.

    Args:
        replay_urls: Dictionary mapping simulation names to lists of replay URLs
        agent_step: Current agent step
        epoch: Current epoch
        wandb_run: The wandb run object
    """
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
            elif sim_name == "replay":
                if "replay" not in replay_groups:
                    replay_groups["replay"] = []
                replay_groups["replay"].extend(urls)
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
                # Single episode - just show the link
                player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={urls[0]}"
                links.append(f'<a href="{player_url}" target="_blank">{name}</a>')
            else:
                # Multiple episodes - show with numbers
                episode_links = []
                for i, url in enumerate(urls, 1):
                    player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={url}"
                    episode_links.append(f'<a href="{player_url}" target="_blank">{i}</a>')
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Join all links with " | " separator and add epoch prefix
        html_content = f"epoch {epoch}: " + " | ".join(links)
    else:
        html_content = f"epoch {epoch}: No replays available."

    # Log the unified HTML with step parameter for wandb's epoch slider
    link_summary = {"replays/all_links": wandb.Html(html_content)}
    wandb_run.log(link_summary, step=agent_step)
