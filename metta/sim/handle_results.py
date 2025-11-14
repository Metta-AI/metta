import logging
from collections import defaultdict
from typing import Any

import wandb
from rich.console import Console
from rich.table import Table

from metta.common.util.collections import remove_none_keys
from metta.common.util.constants import METTASCOPE_REPLAY_URL_PREFIX
from metta.common.util.log_config import get_console, should_use_rich_console
from metta.common.wandb.context import WandbRun
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.rl.wandb import (
    POLICY_EVALUATOR_EPOCH_METRIC,
    POLICY_EVALUATOR_METRIC_PREFIX,
    POLICY_EVALUATOR_STEP_METRIC,
    setup_policy_evaluator_metrics,
)
from metta.sim.runner import SimulationRunResult
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries

logger = logging.getLogger(__name__)


def get_replay_html_payload(
    replay_urls: dict[str, list[str]],
) -> dict[str, Any]:
    """Upload organized replay HTML links to wandb."""
    if not replay_urls:
        return {}
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
    return remove_none_keys({"replays/all": wandb.Html(html_content)})


def _form_mettascope_link(url: str, name: str) -> str:
    return f'<a href="{METTASCOPE_REPLAY_URL_PREFIX}{url}" target="_blank">{name}</a>'


# This gets the sim results into a format we know how to submit to wandb
# We should move away from this towards something with a schema that doesn't give e.g. `category` and `sim_name` meaning
def to_eval_results(
    rollout_results: list[SimulationRunResult], target_policy_idx: int, num_policies: int
) -> EvalResults:
    summaries = build_multi_episode_rollout_summaries(
        rollout_results=[result.results for result in rollout_results], num_policies=num_policies
    )
    simulation_scores: dict[tuple[str, str], float] = {}
    category_scores_accum: defaultdict[str, list[float]] = defaultdict(list)
    replay_urls: dict[str, list[str]] = {}

    for i, (result, summary) in enumerate(zip(rollout_results, summaries, strict=True)):
        category = result.run.episode_tags.get("category", "unknown")
        sim_name = result.run.episode_tags.get("name", f"unknown_{i}")
        policy_rewards: list[float] = []
        for per_policy_rewards in summary.per_episode_per_policy_avg_rewards.values():
            if not per_policy_rewards or len(per_policy_rewards) <= target_policy_idx:
                continue
            policy_reward = per_policy_rewards[target_policy_idx]
            if policy_reward is not None:
                policy_rewards.append(float(policy_reward))

        avg_reward = sum(policy_rewards) / len(policy_rewards) if policy_rewards else 0.0
        simulation_scores[(category, sim_name)] = avg_reward
        category_scores_accum[category].append(avg_reward)

        if result.replay_urls:
            replay_urls[f"{category}.{sim_name}"] = list(result.replay_urls.values())

    category_scores = {
        category: sum(values) / len(values) for category, values in category_scores_accum.items() if values
    }

    return EvalResults(
        scores=EvalRewardSummary(
            category_scores=category_scores,
            simulation_scores=simulation_scores,
        ),
        replay_urls=replay_urls,
    )


def send_eval_results_to_wandb(
    *,
    rollout_results: list[SimulationRunResult],
    epoch: int,
    agent_step: int,
    wandb_run: WandbRun,
    during_training: bool = False,
    should_finish_run: bool = False,
) -> None:
    eval_results = to_eval_results(rollout_results, num_policies=1, target_policy_idx=0)

    # Get metrics payload
    metrics_to_log: dict[str, float] = {
        f"{POLICY_EVALUATOR_METRIC_PREFIX}/eval_{k}": v
        for k, v in eval_results.scores.to_wandb_metrics_format().items()
    }
    metrics_to_log.update(
        {
            f"overview/{POLICY_EVALUATOR_METRIC_PREFIX}/{category}_score": score
            for category, score in eval_results.scores.category_scores.items()
        }
    )
    metrics_to_log.update(get_replay_html_payload(eval_results.replay_urls))
    if not during_training:
        try:
            setup_policy_evaluator_metrics(wandb_run)
        except Exception:
            logger.warning("Failed to set default axes for policy evaluator metrics. Continuing")
            pass
        metrics_to_log.update({POLICY_EVALUATOR_STEP_METRIC: agent_step, POLICY_EVALUATOR_EPOCH_METRIC: epoch})
        wandb_run.log(metrics_to_log)
    else:
        wandb_run.log(metrics_to_log, step=epoch)
    if should_finish_run:
        wandb_run.finish()


def _truncate_name(name: str, max_length: int = 60) -> str:
    if len(name) <= max_length:
        return name
    if "/" in name:
        return name.split("/")[-1][-max_length:]
    return name[-max_length:]


def render_eval_summary(rollout_results: list[SimulationRunResult], policy_names: list[str]) -> None:
    policy_names = [_truncate_name(name) for name in policy_names]
    summaries = build_multi_episode_rollout_summaries(
        [result.results for result in rollout_results], num_policies=len(policy_names)
    )
    names = [
        f"{result.run.episode_tags.get('category', 'unknown')}.{result.run.episode_tags.get('name', f'unknown_{i}')}"
        for i, result in enumerate(rollout_results)
    ]
    mission_summaries = list(zip(names, summaries, strict=True))

    def _print(content) -> None:
        if should_use_rich_console():
            get_console().print(content)
        else:
            console = Console(record=True)
            console.print(content)
            logger.info("\n" + console.export_text())

    if len(policy_names) > 1:
        _print("\n[bold cyan]Policy Assignments[/bold cyan]")
        assignment_table = Table(show_header=True, header_style="bold magenta")
        assignment_table.add_column("Simulation")
        assignment_table.add_column("Policy")
        assignment_table.add_column("Num Agents", justify="right")
        for mission_name, mission in mission_summaries:
            for policy_idx, policy_summary in enumerate(mission.policy_summaries):
                assignment_table.add_row(
                    mission_name,
                    policy_names[policy_idx],
                    str(policy_summary.agent_count),
                )
        _print(assignment_table)

    _print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Simulation")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for mission_name, mission in mission_summaries:
        for key, value in mission.avg_game_stats.items():
            game_stats_table.add_row(mission_name, key, f"{value:.2f}")
    _print(game_stats_table)

    if any(policy.action_timeouts for mission in summaries for policy in mission.policy_summaries):
        _print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Simulation")
        timeouts_table.add_column("Policy")
        timeouts_table.add_column("Timeouts", justify="right")
        for mission_name, mission in mission_summaries:
            for i, policy_summary in enumerate(mission.policy_summaries):
                if policy_summary.action_timeouts > 0:
                    timeouts_table.add_row(
                        mission_name,
                        policy_names[i],
                        str(policy_summary.action_timeouts),
                    )
        _print(timeouts_table)

    _print("\n[bold cyan]Average Policy Stats[/bold cyan]")
    for i, policy_name in enumerate(policy_names):
        policy_table = Table(title=policy_name, show_header=True, header_style="bold magenta")
        policy_table.add_column("Mission")
        policy_table.add_column("Metric")
        policy_table.add_column("Average", justify="right")
        for mission_name, mission in mission_summaries:
            metrics = mission.policy_summaries[i].avg_agent_metrics
            if not metrics:
                policy_table.add_row(mission_name, "-", "-")
                continue
            for key, value in metrics.items():
                policy_table.add_row(mission_name, key, f"{value:.2f}")
        _print(policy_table)

    _print("\n[bold cyan]Average Reward per Agent[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Mission")
    summary_table.add_column("Episode", justify="right")
    for policy_name in policy_names:
        summary_table.add_column(policy_name, justify="right")

    for mission_name, mission in mission_summaries:
        for episode_idx, avg_rewards in sorted(mission.per_episode_per_policy_avg_rewards.items(), key=lambda x: x[0]):
            row = [mission_name, str(episode_idx)]
            row.extend((f"{value:.2f}" if value is not None else "-" for value in avg_rewards))
            summary_table.add_row(*row)

    _print(summary_table)
