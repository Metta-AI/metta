#!/usr/bin/env -S uv run
"""
Standalone script to analyze Social Fluid Intelligence metrics from saved evaluation results.

This script can be used to post-process evaluation results and compute social fluidity metrics.
It expects evaluation results in a format compatible with MultiEpisodeRolloutResult.

Usage:
    uv run python packages/cogames/scripts/analyze_social_fluidity.py --results <path_to_results.json>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from cogames.social_fluidity_analyzer import SocialFluidityAnalyzer
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_results_from_json(filepath: str) -> list[MultiEpisodeRolloutResult]:
    """Load evaluation results from a JSON file.

    Expected format: List of mission results, where each mission result contains
    a list of episodes with assignments, rewards, and stats.

    Args:
        filepath: Path to JSON file containing evaluation results

    Returns:
        List of MultiEpisodeRolloutResult objects
    """
    with open(filepath) as f:
        data = json.load(f)

    # Convert JSON data back to MultiEpisodeRolloutResult objects
    # This assumes the JSON was saved in a compatible format
    # You may need to adjust this based on your actual data format
    results = []
    for mission_data in data:
        episodes = []
        for episode_data in mission_data.get("episodes", []):
            episode = EpisodeRolloutResult(
                assignments=np.array(episode_data["assignments"]),
                rewards=np.array(episode_data["rewards"]),
                action_timeouts=np.array(episode_data.get("action_timeouts", [0] * len(episode_data["rewards"]))),
                stats=episode_data.get("stats", {}),
                replay_path=episode_data.get("replay_path"),
                steps=episode_data.get("steps", 0),
                max_steps=episode_data.get("max_steps", 1000),
            )
            episodes.append(episode)
        results.append(MultiEpisodeRolloutResult(episodes=episodes))

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Social Fluid Intelligence metrics from evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to JSON file containing evaluation results",
    )
    parser.add_argument(
        "--num-policies",
        type=int,
        required=True,
        help="Number of distinct policies in the evaluation",
    )
    parser.add_argument(
        "--risk-parameter",
        type=float,
        default=1.0,
        help="Risk parameter lambda for downside deviation (default: 1.0)",
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=0.1,
        help="Top percentile for specialization calculation (default: 0.1 = top 10%%)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters for team synergy analysis (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save results as CSV",
    )

    args = parser.parse_args()

    console = Console()

    # Load results
    console.print(f"[cyan]Loading results from {args.results}[/cyan]")
    try:
        results = load_results_from_json(args.results)
        console.print(f"[green]Loaded {len(results)} mission results[/green]")
    except Exception as e:
        console.print(f"[red]Error loading results: {e}[/red]")
        return

    # Create analyzer
    analyzer = SocialFluidityAnalyzer(
        rollout_results=results,
        num_policies=args.num_policies,
        risk_parameter=args.risk_parameter,
        top_percentile=args.top_percentile,
        n_clusters=args.n_clusters,
    )

    # Analyze all agents
    console.print("\n[bold cyan]Computing Social Fluid Intelligence Metrics...[/bold cyan]")
    metrics_df = analyzer.analyze_all_agents()

    # Display results
    console.print("\n[bold cyan]Social Fluid Intelligence Metrics[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Policy ID", justify="right")
    table.add_column("Classification")
    table.add_column("Robustness", justify="right")
    table.add_column("Specialization", justify="right")
    table.add_column("Team Synergy", justify="right")
    table.add_column("Mean Score", justify="right")

    for _, row in metrics_df.iterrows():
        policy_idx = int(row["policy_idx"])
        classification = analyzer.classify_agent(policy_idx)
        table.add_row(
            str(policy_idx),
            classification,
            f"{row['robustness']:.3f}",
            f"{row['specialization']:.3f}",
            f"{row['team_synergy_sensitivity']:.3f}",
            f"{row['mean_score']:.3f}",
        )

    console.print(table)

    # Print summary statistics
    console.print("\n[bold cyan]Summary Statistics[/bold cyan]")
    console.print(f"Global Mean Score: {analyzer.global_mean:.3f}")
    console.print(f"Global Variance: {analyzer.global_variance:.3f}")
    console.print(f"Total Games Analyzed: {len(analyzer.df)}")
    console.print(f"Total Episodes: {len(analyzer.df.groupby(['mission_idx', 'episode_idx']))}")

    # Save to CSV if requested
    if args.output:
        metrics_df.to_csv(args.output, index=False)
        console.print(f"\n[green]Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
