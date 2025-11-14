from collections import defaultdict

from rich.table import Table

from metta.common.util.log_config import get_console, should_use_rich_console
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.sim.runner import SimulationRunResult
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries


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


def render_eval_summary(rollout_results: list[SimulationRunResult], policy_names: list[str]) -> None:
    if should_use_rich_console():
        render_rich_eval_summary(rollout_results, policy_names)
    else:
        # TODO: Nishad: render_plain_eval_summary(rollout_results, policy_names)
        pass


def render_rich_eval_summary(rollout_results: list[SimulationRunResult], policy_names: list[str]) -> None:
    summaries = build_multi_episode_rollout_summaries(
        [result.results for result in rollout_results], num_policies=len(policy_names)
    )
    names = [
        f"{result.run.episode_tags.get('category', 'unknown')}.{result.run.episode_tags.get('name', f'unknown_{i}')}"
        for i, result in enumerate(rollout_results)
    ]
    mission_summaries = list(zip(names, summaries, strict=True))

    console = get_console()
    console.print("\n[bold cyan]Policy Assignments[/bold cyan]")
    assignment_table = Table(show_header=True, header_style="bold magenta")
    assignment_table.add_column("Mission")
    assignment_table.add_column("Policy")
    assignment_table.add_column("Num Agents", justify="right")
    for mission_name, mission in mission_summaries:
        for policy_idx, policy_summary in enumerate(mission.policy_summaries):
            assignment_table.add_row(
                mission_name,
                policy_names[policy_idx],
                str(policy_summary.agent_count),
            )
    console.print(assignment_table)

    console.print("\n[bold cyan]Average Policy Stats[/bold cyan]")
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
        console.print(policy_table)

    console.print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Mission")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for mission_name, mission in mission_summaries:
        for key, value in mission.avg_game_stats.items():
            game_stats_table.add_row(mission_name, key, f"{value:.2f}")
    console.print(game_stats_table)

    console.print("\n[bold cyan]Average Reward per Agent[/bold cyan]")
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

    console.print(summary_table)

    if any(policy.action_timeouts for mission in summaries for policy in mission.policy_summaries):
        console.print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Mission")
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
        console.print(timeouts_table)

    # TODO: Nishad: add replay urls here
