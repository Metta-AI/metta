from __future__ import annotations

import json
import logging
import os
import time
import weakref
from typing import Any, Dict, Optional

import torch.nn as nn
import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Use WeakKeyDictionary to associate state with each wandb.Run without mutating the object
_ABORT_STATE: weakref.WeakKeyDictionary[wandb.sdk.wandb_run.Run, Dict[str, Any]] = weakref.WeakKeyDictionary()


def abort_requested(wandb_run: wandb.sdk.wandb_run.Run | None, min_interval_sec: int = 60) -> bool:
    """Return True if the WandB run has an "abort" tag.

    The API call is throttled to *min_interval_sec* seconds.
    """
    if wandb_run is None:
        return False

    state = _ABORT_STATE.setdefault(wandb_run, {"last_check": 0.0, "cached_result": False})
    now = time.time()

    # Return cached result if within throttle interval
    if now - state["last_check"] < min_interval_sec:
        return state["cached_result"]

    # Time to check again
    state["last_check"] = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        state["cached_result"] = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        state["cached_result"] = False

    return state["cached_result"]


def upload_env_configs(env_configs: dict[str, Any], wandb_run: wandb.sdk.wandb_run.Run | None) -> None:
    """Serialize resolved env configs and upload to run files.

    Args:
        env_configs: Dictionary mapping bucket names to their environment configurations
        wandb_run: The wandb run to upload to
    """
    if wandb_run is None:
        return

    try:
        resolved = {k: OmegaConf.to_container(v, resolve=True) for k, v in env_configs.items()}
        payload = json.dumps(resolved, indent=2)
        file_path = os.path.join(wandb_run.dir, "env_configs.json")
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(payload)
        try:
            wandb_run.save(file_path, base_path=wandb_run.dir, policy="now")
        except Exception:
            pass  # offline mode
    except Exception as e:
        logger.warning(f"Failed to upload env configs: {e}")


# Metrics functions moved from metrics.py
def setup_wandb_metrics(wandb_run: Any) -> None:
    """Set up wandb metric definitions for consistent tracking across runs.

    Args:
        wandb_run: The wandb run object
    """
    # Define base metrics
    metrics = ["agent_step", "epoch", "total_time", "train_time"]
    for metric in metrics:
        wandb_run.define_metric(f"metric/{metric}")

    # Set agent_step as the default x-axis for all metrics
    wandb_run.define_metric("*", step_metric="metric/agent_step")

    # Define special metric for reward vs total time
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")


def log_model_parameters(policy: nn.Module, wandb_run: Any) -> None:
    """Log model parameter count to wandb summary.

    Args:
        policy: The policy model
        wandb_run: The wandb run object
    """
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def setup_wandb_metrics_and_log_model(
    policy: nn.Module,
    wandb_run: Optional[Any],
    is_master: bool = True,
) -> None:
    """Convenience function to set up metrics and log model parameters.

    Args:
        policy: The policy model
        wandb_run: The wandb run object (optional)
        is_master: Whether this is the master process in distributed training
    """
    if wandb_run and is_master:
        setup_wandb_metrics(wandb_run)
        log_model_parameters(policy, wandb_run)


def log_training_metrics(
    wandb_run: Any,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Log training metrics to wandb.

    Args:
        wandb_run: The wandb run object
        metrics: Dictionary of metrics to log
        step: The current training step
    """
    wandb_run.log(metrics, step=step)


def define_custom_metric(
    wandb_run: Any,
    metric_name: str,
    step_metric: Optional[str] = None,
) -> None:
    """Define a custom metric with optional step metric.

    Args:
        wandb_run: The wandb run object
        metric_name: Name of the metric to define
        step_metric: Optional step metric to use as x-axis
    """
    if step_metric:
        wandb_run.define_metric(metric_name, step_metric=step_metric)
    else:
        wandb_run.define_metric(metric_name)


# Functions moved from trainer.py
def upload_policy_to_wandb(wandb_run: Any, policy_store: Any, policy_record: Any, force: bool = False) -> Optional[str]:
    """Upload policy to wandb.

    Args:
        wandb_run: The wandb run object
        policy_store: PolicyStore instance
        policy_record: PolicyRecord to upload
        force: Force upload even if not needed

    Returns:
        Artifact name if uploaded, None otherwise
    """
    if not wandb_run or not policy_record:
        return None

    try:
        result = policy_store.add_to_wandb_run(wandb_run.id, policy_record)
        logger.info(f"Uploaded policy to wandb at epoch {policy_record.metadata.get('epoch', 'unknown')}")
        return result
    except Exception as e:
        logger.warning(f"Failed to upload policy to wandb: {e}")
        return None


# Function moved from util/evaluation.py
def upload_replay_html(
    replay_urls: Dict[str, list[str]],
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
                # Single episode - just show the name
                player_url = "https://metta-ai.github.io/metta/?replayUrl=" + urls[0]
                links.append(f'<a href="{player_url}" target="_blank">{name}</a>')
            else:
                # Multiple episodes - show with numbers
                episode_links = []
                for i, url in enumerate(urls, 1):
                    player_url = "https://metta-ai.github.io/metta/?replayUrl=" + url
                    episode_links.append(f'<a href="{player_url}" target="_blank">{i}</a>')
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Join all links with " | " separator and add epoch prefix
        html_content = f"epoch {epoch}: " + " | ".join(links)
    else:
        html_content = f"epoch {epoch}: No replays available."

    # Log the unified HTML with step parameter for wandb's epoch slider
    link_summary = {"replays/all_links": wandb.Html(html_content)}
    wandb_run.log(link_summary, step=agent_step)
