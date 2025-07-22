"""Policy evaluation and replay generation functions."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import wandb

logger = logging.getLogger(__name__)


def evaluate_policy(
    policy_record: Any,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Optional[Any],
    stats_run_id: Optional[Any],
    stats_epoch_start: int,
    epoch: int,
    device: torch.device,
    vectorization: str,
    wandb_policy_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[Any]]:
    """Evaluate policy and return scores.

    Returns:
        Tuple of (eval_scores, stats_epoch_id)
    """
    from metta.common.util.heartbeat import record_heartbeat
    from metta.eval.eval_stats_db import EvalStatsDB
    from metta.sim.simulation_suite import SimulationSuite

    stats_epoch_id = None
    if stats_run_id is not None and stats_client is not None:
        stats_epoch_id = stats_client.create_epoch(
            run_id=stats_run_id,
            start_training_epoch=stats_epoch_start,
            end_training_epoch=epoch,
            attributes={},
        ).id

    logger.info(f"Simulating policy: {policy_record.uri} with config: {sim_suite_config}")

    sim_suite = SimulationSuite(
        config=sim_suite_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        stats_dir="/tmp/stats",
        stats_client=stats_client,
        stats_epoch_id=stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
    )

    result = sim_suite.simulate()
    stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
    logger.info("Simulation complete")

    # Build evaluation metrics
    eval_scores = {}
    categories = set()
    for sim_name in sim_suite_config.simulations.keys():
        categories.add(sim_name.split("/")[0])

    for category in categories:
        score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        record_heartbeat()
        if score is not None:
            eval_scores[f"{category}/score"] = score

    # Get detailed per-simulation scores
    all_scores = stats_db.simulation_scores(policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        eval_scores[f"{category}/{sim_short_name}"] = score

    stats_db.close()
    return eval_scores, stats_epoch_id


def generate_replay(
    policy_record: Any,
    policy_store: Any,
    curriculum: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    replay_dir: str,
    wandb_run: Optional[Any] = None,
) -> None:
    """Generate and upload replay."""
    from metta.sim.simulation import Simulation
    from metta.sim.simulation_config import SingleEnvSimulationConfig

    replay_sim_config = SingleEnvSimulationConfig(
        env="/env/mettagrid/mettagrid",
        num_episodes=1,
        env_overrides=curriculum.get_task().env_cfg(),
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
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={replay_url}"
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
            wandb_run.log(link_summary)

    results.stats_db.close()
    return player_url


def upload_replay_html(
    replay_urls: Dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: Any,
) -> None:
    """Upload replay HTML to wandb with unified view of all replay links."""
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
