"""Replay generation and management functionality for Metta."""

import logging
from typing import Any, Optional

import torch
import wandb
from omegaconf import DictConfig

from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig

logger = logging.getLogger(__name__)


def generate_policy_replay(
    policy_record: Any,
    policy_store: Any,
    trainer_cfg: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    wandb_run: Any | None,
) -> str | None:
    """Generate a replay for the policy.
    
    Args:
        policy_record: Policy record to generate replay for
        policy_store: Policy store for loading policies
        trainer_cfg: Trainer configuration containing curriculum and replay settings
        epoch: Current training epoch
        device: Device to run simulation on
        vectorization: Vectorization mode for simulation
        wandb_run: Optional wandb run for logging
        
    Returns:
        URL to the replay player, or None if no replay was generated
    """
    # Get curriculum from trainer config
    curriculum = curriculum_from_config_path(trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides))

    replay_url = generate_replay(
        policy_record=policy_record,
        policy_store=policy_store,
        curriculum=curriculum,
        epoch=epoch,
        device=device,
        vectorization=vectorization,
        replay_dir=trainer_cfg.simulation.replay_dir,
        wandb_run=wandb_run,
    )

    return replay_url


def generate_replay(
    policy_record: Any,
    policy_store: Any,
    curriculum: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    replay_dir: str,
    wandb_run: Optional[Any] = None,
) -> Optional[str]:
    """Generate and upload replay.
    
    Args:
        policy_record: Policy record to generate replay for
        policy_store: Policy store for loading policies
        curriculum: Curriculum containing the task configuration
        epoch: Current training epoch
        device: Device to run simulation on
        vectorization: Vectorization mode for simulation
        replay_dir: Directory to save replay files
        wandb_run: Optional wandb run for logging
        
    Returns:
        URL to the replay player, or None if no replay was generated
    """
    player_url = None
    # Pass the config as _pre_built_env_config to avoid Hydra loading
    task_cfg = curriculum.get_task().env_cfg()
    replay_sim_config = SingleEnvSimulationConfig(
        env="replay_task",  # Just a descriptive name
        num_episodes=1,
        env_overrides={"_pre_built_env_config": task_cfg},
    )

    replay_simulator = Simulation(
        name=f"replay_{epoch}",
        config=replay_sim_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,
    )

    results = replay_simulator.simulate()

    if wandb_run is not None:
        key, version = results.stats_db.key_and_version(policy_record)
        replay_urls = results.stats_db.get_replay_urls(key, version)
        if len(replay_urls) > 0:
            replay_url = replay_urls[0]
            player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={replay_url}"
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
            wandb_run.log(link_summary)

    results.stats_db.close()
    return player_url


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