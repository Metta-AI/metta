import io
import logging
import uuid
from collections import defaultdict
from typing import Any, NewType

import duckdb
import wandb
from pydantic import Field
from rich.console import Console
from rich.table import Table

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.episode_stats_db import (
    episode_stats_db,
    insert_agent_metric,
    insert_agent_policy,
    insert_episode,
    insert_episode_tag,
    insert_policy_metric,
)
from metta.common.util.collections import remove_none_keys
from metta.common.util.log_config import get_console, should_use_rich_console
from metta.common.wandb.context import WandbRun
from metta.rl.wandb import (
    POLICY_EVALUATOR_EPOCH_METRIC,
    POLICY_EVALUATOR_METRIC_PREFIX,
    POLICY_EVALUATOR_STEP_METRIC,
    setup_policy_evaluator_metrics,
)
from metta.sim.pure_single_episode_runner import PureSingleEpisodeResult
from metta.sim.runner import SimulationRunResult
from mettagrid.base_config import Config
from mettagrid.renderer.mettascope import METTASCOPE_REPLAY_URL_PREFIX
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries
from mettagrid.util.file import http_url

EpisodeId = NewType("EpisodeId", uuid.UUID)

logger = logging.getLogger(__name__)


class EvalRewardSummary(Config):
    category_scores: dict[str, float] = Field(default_factory=dict, description="Average reward for each category")
    simulation_scores: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Average reward for each simulation (category, short_sim_name)"
    )

    @property
    def avg_category_score(self) -> float:
        return sum(self.category_scores.values()) / len(self.category_scores) if self.category_scores else 0

    @property
    def avg_simulation_score(self) -> float:
        return sum(self.simulation_scores.values()) / len(self.simulation_scores) if self.simulation_scores else 0

    def to_wandb_metrics_format(self) -> dict[str, float]:
        return {
            **{f"{category}/score": score for category, score in self.category_scores.items()},
            **{f"{category}/{sim}": score for (category, sim), score in self.simulation_scores.items()},
        }


class EvalResults(Config):
    scores: EvalRewardSummary = Field(..., description="Evaluation scores")
    replay_urls: dict[str, list[str]] = Field(default_factory=dict, description="Replay URLs for each simulation")


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

        replay_paths = [episode.replay_path for episode in result.results.episodes if episode.replay_path]
        if replay_paths:
            replay_urls[f"{category}.{sim_name}"] = [http_url(path) for path in replay_paths]

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
            logger.error("Failed to set default axes for policy evaluator metrics. Continuing", exc_info=True)
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


def render_eval_summary(
    rollout_results: list[SimulationRunResult], policy_names: list[str], verbose: bool = False
) -> None:
    policy_names = [_truncate_name(name) for name in policy_names]
    summaries = build_multi_episode_rollout_summaries(
        [result.results for result in rollout_results], num_policies=len(policy_names)
    )
    sim_names = [
        f"{result.run.episode_tags.get('category', 'unknown')}.{result.run.episode_tags.get('name', f'unknown_{i}')}"
        for i, result in enumerate(rollout_results)
    ]
    sim_summaries = list(zip(sim_names, summaries, strict=True))
    use_rich_console = should_use_rich_console()

    def _print(content) -> None:
        if use_rich_console:
            get_console().print(content)
        else:
            # headless, no stdout, but record everything
            buffer = io.StringIO()
            console = Console(
                record=True,
                file=buffer,
                force_terminal=False,
                color_system=None,
                no_color=True,
            )
            console.print(content)
            text_output = console.export_text()
            logger.info("\n%s", text_output)

    if len(policy_names) > 1 and verbose:
        _print("\n[bold cyan]Policy Assignments[/bold cyan]")
        assignment_table = Table(show_header=True, header_style="bold magenta")
        assignment_table.add_column("Simulation")
        assignment_table.add_column("Policy")
        assignment_table.add_column("Num Agents", justify="right")
        for s_name, s in sim_summaries:
            for policy_idx, policy_summary in enumerate(s.policy_summaries):
                assignment_table.add_row(
                    s_name,
                    policy_names[policy_idx],
                    str(policy_summary.agent_count),
                )
        _print(assignment_table)

    if verbose:
        _print("\n[bold cyan]Average Game Stats[/bold cyan]")
        game_stats_table = Table(show_header=True, header_style="bold magenta")
        game_stats_table.add_column("Simulation")
        game_stats_table.add_column("Metric")
        game_stats_table.add_column("Average", justify="right")
        for s_name, s in sim_summaries:
            for key, value in s.avg_game_stats.items():
                game_stats_table.add_row(s_name, key, f"{value:.2f}")
        _print(game_stats_table)

    if verbose and any(policy.action_timeouts for s in summaries for policy in s.policy_summaries):
        _print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Simulation")
        timeouts_table.add_column("Policy")
        timeouts_table.add_column("Timeouts", justify="right")
        for s_name, s in sim_summaries:
            for i, policy_summary in enumerate(s.policy_summaries):
                if policy_summary.action_timeouts > 0:
                    timeouts_table.add_row(
                        s_name,
                        policy_names[i],
                        str(policy_summary.action_timeouts),
                    )
        _print(timeouts_table)

    if verbose:
        _print("\n[bold cyan]Average Policy Stats[/bold cyan]")
        for i, policy_name in enumerate(policy_names):
            policy_table = Table(title=policy_name, show_header=True, header_style="bold magenta")
            policy_table.add_column("Simulation")
            policy_table.add_column("Metric")
            policy_table.add_column("Average", justify="right")
            for s_name, s in sim_summaries:
                metrics = s.policy_summaries[i].avg_agent_metrics
                if not metrics:
                    policy_table.add_row(s_name, "-", "-")
                    continue
                for key, value in metrics.items():
                    policy_table.add_row(s_name, key, f"{value:.2f}")
            _print(policy_table)

    _print("\n[bold cyan]Average Per-Agent Reward [/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Simulation")
    summary_table.add_column("Episode", justify="right")
    for policy_name in policy_names:
        summary_table.add_column(policy_name, justify="right")

    for s_name, s in sim_summaries:
        for episode_idx, avg_rewards in sorted(s.per_episode_per_policy_avg_rewards.items(), key=lambda x: x[0]):
            row = [s_name, str(episode_idx)]
            row.extend((f"{value:.2f}" if value is not None else "-" for value in avg_rewards))
            summary_table.add_row(*row)

    _print(summary_table)

    replay_rows: list[tuple[str, str, str]] = []
    for s_name, result in zip(sim_names, rollout_results, strict=True):
        replay_paths = [episode.replay_path for episode in result.results.episodes]
        for episode_idx, replay_path in enumerate(replay_paths):
            if replay_path:
                viewer_url = f"{METTASCOPE_REPLAY_URL_PREFIX}{http_url(replay_path)}"
                replay_rows.append((s_name, str(episode_idx), viewer_url))

    if replay_rows:
        if use_rich_console:
            _print("\n[bold cyan]Replay Links[/bold cyan]")
            replay_table = Table(show_header=True, header_style="bold magenta")
            replay_table.add_column("Simulation")
            replay_table.add_column("Episode")
            replay_table.add_column("Viewer")
            for s_name, episode_id, viewer_url in replay_rows:
                replay_table.add_row(s_name, episode_id, f"[link={viewer_url}]Open in Mettascope[/link]")
            _print(replay_table)
        else:
            lines = ["Replay Links:"]
            for s_name, episode_id, viewer_url in replay_rows:
                lines.append(f"- Simulation: {s_name}")
                lines.append(f"  Episode: {episode_id}")
                lines.append(f"  Viewer: {viewer_url}")
            logger.info("\n%s", "\n".join(lines))


# NOTE: This will be removed when we switch all evaluations to the single-episode runner
def write_eval_results_to_observatory(
    *,
    policy_version_ids: list[str],
    rollout_results: list[SimulationRunResult],
    stats_client: StatsClient,
    primary_policy_version_id: str | None = None,
) -> None:
    try:
        with episode_stats_db() as (conn, duckdb_path):
            for sim_result in rollout_results:
                sim_config = sim_result.run
                results = sim_result.results

                for e in results.episodes:
                    episode_id = str(uuid.uuid4())

                    insert_episode(
                        conn,
                        episode_id=episode_id,
                        primary_pv_id=primary_policy_version_id,
                        replay_url=e.replay_path,
                        thumbnail_url=None,
                        eval_task_id=None,
                    )

                    for key, value in sim_config.episode_tags.items():
                        insert_episode_tag(conn, episode_id, key, value)

                    for agent_id, policy_idx in enumerate(e.assignments):
                        pv_id = policy_version_ids[policy_idx]

                        insert_agent_policy(conn, episode_id, pv_id, agent_id)

                        insert_agent_metric(conn, episode_id, agent_id, "reward", float(e.rewards[agent_id]))
                        agent_metrics = e.stats["agent"][agent_id]
                        for metric_name, metric_value in agent_metrics.items():
                            insert_agent_metric(conn, episode_id, agent_id, metric_name, metric_value)

                    policy_failure_steps: dict[int, int] = {}
                    for agent_policy_idx, failure_step in zip(e.assignments, e.failure_steps or (), strict=False):
                        if failure_step is None:
                            continue
                        pidx = int(agent_policy_idx)
                        fstep = int(failure_step)
                        policy_failure_steps[pidx] = min(fstep, policy_failure_steps.get(pidx, fstep))

                    assigned_policy_indices = {int(idx) for idx in e.assignments}
                    for policy_idx in assigned_policy_indices:
                        pv_id = policy_version_ids[policy_idx]
                        failure_step = policy_failure_steps.get(policy_idx)
                        exception_flag = 1.0 if failure_step is not None else 0.0
                        insert_policy_metric(conn, episode_id, pv_id, "exception_flag", exception_flag)
                        if failure_step is not None:
                            insert_policy_metric(conn, episode_id, pv_id, "exception_step", float(failure_step))

            conn.execute("CHECKPOINT")
            logger.info(f"Uploading evaluation results to observatory (DuckDB size: {duckdb_path})")
            response = stats_client.bulk_upload_episodes(str(duckdb_path))
            logger.info(
                f"Successfully uploaded {response.episodes_created} episodes to observatory at {response.duckdb_s3_uri}"
            )
    except Exception as e:
        logger.warning(f"Failed to write evaluation results to observatory: {e}", exc_info=True)


def populate_single_episode_duckdb(
    conn: duckdb.DuckDBPyConnection,
    *,
    episode_tags: dict[str, str],
    policy_version_ids: list[uuid.UUID | None],
    replay_uri: str | None,
    assignments: list[int],
    results: PureSingleEpisodeResult,
) -> EpisodeId:
    episode_id = EpisodeId(uuid.uuid4())

    insert_episode(
        conn,
        episode_id=str(episode_id),
        primary_pv_id=str(policy_version_ids[0]) if policy_version_ids[0] else None,
        replay_url=http_url(replay_uri) if replay_uri else None,
        thumbnail_url=None,
        eval_task_id=None,
    )

    for key, value in episode_tags.items():
        insert_episode_tag(conn, str(episode_id), key, value)

    for agent_id, assignment in enumerate(assignments):
        policy_version_id = policy_version_ids[assignment]
        if policy_version_id:
            insert_agent_policy(conn, str(episode_id), str(policy_version_id), agent_id)

        insert_agent_metric(conn, str(episode_id), agent_id, "reward", results.rewards[agent_id])
        insert_agent_metric(conn, str(episode_id), agent_id, "action_timeout", float(results.action_timeouts[agent_id]))

        agent_stats = results.stats["agent"][agent_id]
        for metric_name, metric_value in agent_stats.items():
            insert_agent_metric(conn, str(episode_id), agent_id, metric_name, metric_value)

    return episode_id


def write_single_episode_to_observatory(
    *,
    episode_tags: dict[str, str],
    policy_version_ids: list[uuid.UUID | None],
    replay_uri: str | None,
    assignments: list[int],
    results: PureSingleEpisodeResult,
    stats_client: StatsClient,
) -> EpisodeId:
    with episode_stats_db() as (conn, db_path):
        episode_id = populate_single_episode_duckdb(
            conn,
            episode_tags=episode_tags,
            policy_version_ids=policy_version_ids,
            replay_uri=replay_uri,
            assignments=assignments,
            results=results,
        )
        conn.execute("CHECKPOINT")
        response = stats_client.bulk_upload_episodes(str(db_path))
        logger.info(f"Uploaded episode: {response.episodes_created} episodes at {response.duckdb_s3_uri}")
    return episode_id
