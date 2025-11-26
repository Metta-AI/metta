#!/usr/bin/env python3
"""Generate plots from curriculum training analysis JSON files."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use("default")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True


def _get_mission_base_name(mission_variant: str) -> str:
    """Extract base mission name from mission:variant format."""
    if ":" in mission_variant:
        return mission_variant.split(":")[0]
    return mission_variant


def _get_mission_suite(mission_base: str) -> str:
    """Categorize mission into a suite."""
    if mission_base.startswith("diagnostic_"):
        return "Diagnostic"
    elif mission_base in ["harvest", "repair", "vibe_check", "assemble", "unclip_drills", "signs_and_portents"]:
        return "Training Facility"
    elif mission_base.startswith("extractor_hub"):
        return "Extractor Hub"
    elif mission_base in [
        "go_together",
        "oxygen_bottleneck",
        "energy_starved",
        "collect_resources_classic",
        "collect_resources_spread",
        "collect_far",
        "divide_and_conquer",
        "single_use_swarm",
    ]:
        return "Eval Missions"
    else:
        return "Other"


def load_analysis_files(json_dir: Path) -> list[dict[str, Any]]:
    """Load all analysis JSON files."""
    analyses = []
    for json_file in sorted(json_dir.glob("analysis_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                analyses.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
    return analyses


def load_wandb_data(wandb_data_dir: Path) -> dict[str, dict[str, Any]]:
    """Load cached W&B data."""
    combined_file = wandb_data_dir / "all_runs.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            return json.load(f)
    return {}


def plot_per_env_metrics(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot per-environment chest deposits and reward."""
    # Collect all environments across all runs and aggregate metrics
    env_rewards: dict[str, list[float]] = defaultdict(list)
    env_deposits: dict[str, list[float]] = defaultdict(list)

    for run_data in wandb_data.values():
        per_env = run_data.get("per_env_metrics", {})
        for env_name, env_data in per_env.items():
            # Skip .avg entries
            if ".avg" in env_name:
                continue
            reward = env_data.get("avg_reward", 0.0)
            deposits = env_data.get("chest_deposits_avg", env_data.get("chest_deposits_rate", 0.0))
            if reward > 0 or deposits > 0:
                env_rewards[env_name].append(reward)
                env_deposits[env_name].append(deposits)

    # Calculate averages per environment
    env_avg_rewards = {}
    env_avg_deposits = {}
    for env in env_rewards.keys():
        env_avg_rewards[env] = np.mean(env_rewards[env]) if env_rewards[env] else 0.0
        env_avg_deposits[env] = np.mean(env_deposits[env]) if env_deposits[env] else 0.0

    # Sort by reward (or could sort by deposits)
    sorted_envs = sorted(env_avg_rewards.items(), key=lambda x: x[1], reverse=True)

    # Take top 50 environments for readability
    top_envs = sorted_envs[:50]
    env_names = [e[0] for e in top_envs]
    avg_rewards = [env_avg_rewards[e[0]] for e in top_envs]
    avg_deposits = [env_avg_deposits[e[0]] for e in top_envs]

    # Create side-by-side bar chart
    x = np.arange(len(env_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 10))
    bars1 = ax.bar(
        x - width / 2, avg_rewards, width, label="Average Reward", color="steelblue", alpha=0.8, edgecolor="black"
    )
    bars2 = ax.bar(
        x + width / 2, avg_deposits, width, label="Average Chest Deposits", color="orange", alpha=0.8, edgecolor="black"
    )

    ax.set_xlabel("Environment", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title("Average Reward and Chest Deposits per Environment (Top 50)", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=90, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars (only for non-zero values to avoid clutter)
    for bar, val in zip(bars1, avg_rewards, strict=True):
        if val > 0.1:  # Only label if significant
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    for bar, val in zip(bars2, avg_deposits, strict=True):
        if val > 0.01:  # Only label if significant
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "per_env_reward_deposits.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved per_env_reward_deposits.png")

    # Also keep the heatmaps for detailed per-run analysis
    runs = [analysis.get("run_id", "unknown") for analysis in analyses]
    base_envs = sorted(env_avg_rewards.keys())[:30]  # Top 30 for heatmaps

    # Build matrices: run x env -> metric value
    reward_matrix = []
    deposits_matrix = []

    for run_id in runs:
        run_data = wandb_data.get(run_id, {})
        per_env = run_data.get("per_env_metrics", {})

        reward_row = []
        deposits_row = []

        for env in base_envs:
            env_data = per_env.get(env, {})
            reward_row.append(env_data.get("avg_reward", 0.0))
            deposits_row.append(env_data.get("chest_deposits_avg", env_data.get("chest_deposits_rate", 0.0)))

        reward_matrix.append(reward_row)
        deposits_matrix.append(deposits_row)

    # Plot 1: Per-env reward heatmap
    if reward_matrix:
        fig, ax = plt.subplots(figsize=(max(14, len(base_envs) * 0.4), max(8, len(runs) * 0.3)))
        reward_array = np.array(reward_matrix)

        im = ax.imshow(reward_array, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(base_envs)))
        ax.set_yticks(np.arange(len(runs)))
        ax.set_xticklabels(base_envs, rotation=90, ha="right", fontsize=7)
        ax.set_yticklabels(runs, fontsize=6)
        ax.set_xlabel("Environment", fontweight="bold")
        ax.set_ylabel("Run ID", fontweight="bold")
        ax.set_title("Average Reward per Environment", fontweight="bold", fontsize=12)
        plt.colorbar(im, ax=ax, label="Average Reward")
        plt.tight_layout()
        plt.savefig(output_dir / "per_env_reward_heatmap.png", bbox_inches="tight")
        plt.close()
        print("✓ Saved per_env_reward_heatmap.png")

    # Plot 2: Per-env chest deposits heatmap
    if deposits_matrix:
        fig, ax = plt.subplots(figsize=(max(14, len(base_envs) * 0.4), max(8, len(runs) * 0.3)))
        deposits_array = np.array(deposits_matrix)

        im = ax.imshow(deposits_array, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(np.arange(len(base_envs)))
        ax.set_yticks(np.arange(len(runs)))
        ax.set_xticklabels(base_envs, rotation=90, ha="right", fontsize=7)
        ax.set_yticklabels(runs, fontsize=6)
        ax.set_xlabel("Environment", fontweight="bold")
        ax.set_ylabel("Run ID", fontweight="bold")
        ax.set_title("Average Chest Deposits per Environment", fontweight="bold", fontsize=12)
        plt.colorbar(im, ax=ax, label="Avg Deposits")
        plt.tight_layout()
        plt.savefig(output_dir / "per_env_deposits_heatmap.png", bbox_inches="tight")
        plt.close()
        print("✓ Saved per_env_deposits_heatmap.png")


def plot_comparison_summary(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot separate comparisons for each metric across policies."""
    runs = []
    heart_deposits = []
    heart_gains = []
    avg_rewards = []
    episodes = []
    agent_steps = []

    print("Extracting metrics from cached W&B data...")
    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        runs.append(run_id)

        # Get metrics from cached W&B data
        run_data = wandb_data.get(run_id, {})
        global_metrics = run_data.get("global_metrics", {})

        heart_gains.append(global_metrics.get("heart_gained_rate", 0.0))
        heart_deposits.append(global_metrics.get("chest_deposited_rate", 0.0))
        avg_rewards.append(global_metrics.get("avg_reward", 0.0))

        episodes.append(analysis.get("total_episodes", 0))
        agent_steps.append(analysis.get("agent_steps", 0) / 1e9)  # Convert to billions

    x = np.arange(len(runs))

    # Plot 1: Chest Deposits Rate (game-level metric)
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, heart_deposits, color="orange", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Policy", fontweight="bold")
    ax.set_ylabel("Chest Deposits Rate", fontweight="bold")
    ax.set_title("Chest Deposits Rate by Policy\n(% of episodes with deposits)", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, heart_deposits, strict=True):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_chest_deposits.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_chest_deposits.png")

    # Plot 2: Total episodes
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, episodes, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Run ID", fontweight="bold")
    ax.set_ylabel("Total Episodes", fontweight="bold")
    ax.set_title("Training Episodes", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(episodes):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_episodes.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_episodes.png")

    # Plot 3: Agent steps (billions)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, agent_steps, color="purple", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Run ID", fontweight="bold")
    ax.set_ylabel("Agent Steps (Billions)", fontweight="bold")
    ax.set_title("Training Progress", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(agent_steps):
        if v > 0:
            ax.text(i, v, f"{v:.1f}B", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_agent_steps.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_agent_steps.png")

    # Plot 4: Final Reward (most recent reward value per policy)
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, avg_rewards, color="green", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Policy", fontweight="bold")
    ax.set_ylabel("Final Reward", fontweight="bold")
    ax.set_title("Final Reward by Policy\n(Most Recent Reward Value)", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, total_rewards, strict=True):
        if val > 0:
            # Format large numbers nicely
            if val >= 1e6:
                label = f"{val / 1e6:.1f}M"
            elif val >= 1e3:
                label = f"{val / 1e3:.1f}K"
            else:
                label = f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, val, label, ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_total_reward.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_total_reward.png")


def plot_heart_gained_comparison(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot comparison of average heart gained rate across runs."""
    runs = []
    heart_gains = []

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        runs.append(run_id)

        run_data = wandb_data.get(run_id, {})
        global_metrics = run_data.get("global_metrics", {})
        heart_gains.append(global_metrics.get("heart_gained_rate", 0.0))

    x = np.arange(len(runs))

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, heart_gains, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Run ID", fontweight="bold")
    ax.set_ylabel("Heart Gained Rate", fontweight="bold")
    ax.set_title("Average Heart Gained Rate by Run", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, heart_gains, strict=True):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "heart_gained_comparison.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved heart_gained_comparison.png")


def plot_average_reward_per_mission(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot average reward per mission across all runs using W&B data."""
    # Collect all missions and their rewards from per-env metrics
    mission_rewards: dict[str, list[float]] = defaultdict(list)

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        run_data = wandb_data.get(run_id, {})
        per_env = run_data.get("per_env_metrics", {})

        for env_name, env_data in per_env.items():
            # Skip .avg entries, use base mission names
            if ".avg" in env_name:
                continue
            reward = env_data.get("avg_reward", 0.0)
            if reward > 0:
                mission_rewards[env_name].append(reward)

    if not mission_rewards:
        print("⚠ No mission reward data found")
        return

    # Calculate average reward per mission
    mission_avg_rewards = {}
    for mission, rewards in mission_rewards.items():
        mission_avg_rewards[mission] = np.mean(rewards)

    # Sort by average reward
    sorted_missions = sorted(mission_avg_rewards.items(), key=lambda x: x[1], reverse=True)

    # Take top 50 missions for readability
    top_missions = sorted_missions[:50]
    mission_names = [m[0] for m in top_missions]
    avg_rewards = [m[1] for m in top_missions]

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(range(len(mission_names)), avg_rewards, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(mission_names)))
    ax.set_yticklabels(mission_names, fontsize=8)
    ax.set_xlabel("Average Reward", fontweight="bold")
    ax.set_ylabel("Mission", fontweight="bold")
    ax.set_title("Top 50 Missions by Average Reward", fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (_bar, reward) in enumerate(zip(bars, avg_rewards, strict=True)):
        ax.text(reward, i, f"{reward:.2f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "average_reward_per_mission.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved average_reward_per_mission.png")


def plot_heart_deposits_per_mission(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot average heart deposits rate per mission across all runs."""
    # Collect all missions and their deposit rates from per-env metrics
    mission_deposits: dict[str, list[float]] = defaultdict(list)

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        run_data = wandb_data.get(run_id, {})
        per_env = run_data.get("per_env_metrics", {})

        for env_name, env_data in per_env.items():
            # Skip .avg entries, use base mission names
            if ".avg" in env_name:
                continue
            deposits_avg = env_data.get("chest_deposits_avg", env_data.get("chest_deposits_rate", 0.0))
            if deposits_avg >= 0:  # Include 0 values too
                mission_deposits[env_name].append(deposits_avg)

    if not mission_deposits:
        print("⚠ No mission deposit data found")
        return

    # Calculate average deposit rate per mission
    mission_avg_deposits = {}
    for mission, deposits in mission_deposits.items():
        mission_avg_deposits[mission] = np.mean(deposits)

    # Sort by average deposit rate
    sorted_missions = sorted(mission_avg_deposits.items(), key=lambda x: x[1], reverse=True)

    # Take top 50 missions for readability
    top_missions = sorted_missions[:50]
    mission_names = [m[0] for m in top_missions]
    avg_deposits = [m[1] for m in top_missions]

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(range(len(mission_names)), avg_deposits, color="orange", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(mission_names)))
    ax.set_yticklabels(mission_names, fontsize=8)
    ax.set_xlabel("Average Chest Deposits per Episode", fontweight="bold")
    ax.set_ylabel("Mission", fontweight="bold")
    ax.set_title("Top 50 Missions by Average Chest Deposits per Episode", fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (_bar, deposits) in enumerate(zip(bars, avg_deposits, strict=True)):
        ax.text(deposits, i, f"{deposits:.3f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "average_deposits_per_mission.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved average_deposits_per_mission.png")


def plot_heart_gained_per_mission(
    analyses: list[dict[str, Any]], output_dir: Path, wandb_data: dict[str, dict[str, Any]]
) -> None:
    """Plot average heart gained rate per mission across all runs."""
    # Note: We don't have per-mission heart.gained in the W&B data
    # We only have global heart_gained_rate, so we'll use that as a baseline
    # and show which missions are present in runs with high heart gained rates

    # Collect missions and their associated global heart gained rates
    mission_heart_rates: dict[str, list[float]] = defaultdict(list)

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        run_data = wandb_data.get(run_id, {})
        global_metrics = run_data.get("global_metrics", {})
        heart_gained_rate = global_metrics.get("heart_gained_rate", 0.0)

        # Get all missions from this run
        per_env = run_data.get("per_env_metrics", {})
        for env_name in per_env.keys():
            if ".avg" not in env_name:
                mission_heart_rates[env_name].append(heart_gained_rate)

    if not mission_heart_rates:
        print("⚠ No mission heart gained data found")
        return

    # Calculate average heart gained rate per mission (using global rate from runs that include this mission)
    mission_avg_heart_rates = {}
    for mission, rates in mission_heart_rates.items():
        mission_avg_heart_rates[mission] = np.mean(rates)

    # Sort by average heart gained rate
    sorted_missions = sorted(mission_avg_heart_rates.items(), key=lambda x: x[1], reverse=True)

    # Take top 50 missions for readability
    top_missions = sorted_missions[:50]
    mission_names = [m[0] for m in top_missions]
    avg_heart_rates = [m[1] for m in top_missions]

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(range(len(mission_names)), avg_heart_rates, color="red", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(mission_names)))
    ax.set_yticklabels(mission_names, fontsize=8)
    ax.set_xlabel("Average Heart Gained Rate (Global)", fontweight="bold")
    ax.set_ylabel("Mission", fontweight="bold")
    title = "Top 50 Missions by Average Heart Gained Rate\n(Note: Uses global rate from runs containing each mission)"
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (_bar, rate) in enumerate(zip(bars, avg_heart_rates, strict=True)):
        ax.text(rate, i, f"{rate:.3f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "average_heart_gained_per_mission.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved average_heart_gained_per_mission.png")


def plot_mission_heatmap_by_suite(analyses: list[dict[str, Any]], output_dir: Path) -> None:
    """Create heatmaps of mission performance grouped by mission suite."""
    # Collect all missions and categorize by suite
    suite_missions: dict[str, set[str]] = defaultdict(set)
    mission_data: dict[tuple[str, str], dict[str, Any]] = {}  # (run_id, mission) -> data

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")

        # Collect missions from all categories
        for mission in analysis.get("missions_mastered", []):
            base = _get_mission_base_name(mission)
            suite = _get_mission_suite(base)
            suite_missions[suite].add(base)
            mission_data[(run_id, mission)] = {"status": "mastered", "base": base, "suite": suite}

        for stuck in analysis.get("missions_stuck", []):
            if isinstance(stuck, dict):
                mission = stuck.get("mission", "")
                base = _get_mission_base_name(mission)
                suite = _get_mission_suite(base)
                suite_missions[suite].add(base)
                mission_data[(run_id, mission)] = {
                    "status": "stuck",
                    "base": base,
                    "suite": suite,
                    "reward": stuck.get("avg_reward", 0.0),
                }

        for mission in analysis.get("missions_failing", []):
            base = _get_mission_base_name(mission)
            suite = _get_mission_suite(base)
            suite_missions[suite].add(base)
            mission_data[(run_id, mission)] = {"status": "failing", "base": base, "suite": suite}

    # Create a heatmap for each suite
    runs = sorted(set(analysis.get("run_id", "unknown") for analysis in analyses))

    for suite, missions in suite_missions.items():
        if not missions:
            continue

        missions_list = sorted(missions)
        # Limit to 30 missions per suite for readability
        if len(missions_list) > 30:
            missions_list = missions_list[:30]

        # Build matrix: run_id x mission -> status (1=mastered, 0.5=stuck, 0=failing, -1=not present)
        matrix = []
        for run_id in runs:
            row = []
            for mission_base in missions_list:
                # Find matching missions for this run and base
                matching = [
                    (r, m, d) for (r, m), d in mission_data.items() if r == run_id and d["base"] == mission_base
                ]

                if not matching:
                    row.append(-1.0)  # Not present
                else:
                    # Use the first matching mission's status
                    _, _, data = matching[0]
                    if data["status"] == "mastered":
                        row.append(1.0)
                    elif data["status"] == "stuck":
                        row.append(0.5)
                    else:  # failing
                        row.append(0.0)
            matrix.append(row)

        if not matrix or not missions_list:
            continue

        # Create heatmap
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(max(12, len(missions_list) * 0.5), max(6, len(runs) * 0.3)))
        matrix_array = np.array(matrix)

        colors = ["gray", "red", "orange", "green"]
        cmap = ListedColormap(colors)

        ax.imshow(matrix_array, aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5)

        ax.set_xticks(np.arange(len(missions_list)))
        ax.set_yticks(np.arange(len(runs)))
        ax.set_xticklabels(missions_list, rotation=90, ha="right", fontsize=7)
        ax.set_yticklabels(runs, fontsize=6)
        ax.set_xlabel("Mission", fontweight="bold")
        ax.set_ylabel("Run ID", fontweight="bold")
        title = (
            f"Mission Performance Heatmap - {suite}\n"
            "(Green=Reward, Orange=Low Heart Deposits, Red=Low Heart Gains, Gray=Not Present)"
        )
        ax.set_title(title, fontweight="bold", fontsize=12)

        plt.tight_layout()
        safe_suite_name = suite.replace(" ", "_").lower()
        plt.savefig(output_dir / f"mission_heatmap_{safe_suite_name}.png", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved mission_heatmap_{safe_suite_name}.png")


def plot_hearts_obtained_vs_deposited(analyses: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot hearts obtained vs deposited rates for stuck missions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hearts Obtained vs Deposited Analysis", fontsize=16, fontweight="bold")

    all_stuck = []
    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        for stuck in analysis.get("missions_stuck", []):
            if isinstance(stuck, dict):
                all_stuck.append(
                    {
                        "run_id": run_id,
                        "mission": stuck.get("mission", "unknown"),
                        "hearts_obtained": stuck.get("hearts_obtained_rate", 0),
                        "hearts_deposited": stuck.get("hearts_deposited_rate", 0),
                    }
                )

    if not all_stuck:
        print("⚠ No stuck missions found for hearts analysis")
        return

    df = pd.DataFrame(all_stuck)

    # Plot 1: Scatter plot
    axes[0, 0].scatter(
        df["hearts_obtained"], df["hearts_deposited"], alpha=0.6, s=100, edgecolors="black", linewidth=0.5
    )
    axes[0, 0].axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Stuck threshold (20%)")
    axes[0, 0].axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Obtained threshold (50%)")
    axes[0, 0].set_xlabel("Hearts Obtained Rate", fontweight="bold")
    axes[0, 0].set_ylabel("Hearts Deposited Rate", fontweight="bold")
    axes[0, 0].set_title("Hearts Obtained vs Deposited (All Stuck Missions)", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Distribution of hearts obtained
    axes[0, 1].hist(df["hearts_obtained"], bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    mean_obtained = df["hearts_obtained"].mean()
    axes[0, 1].axvline(x=mean_obtained, color="red", linestyle="--", label=f"Mean: {mean_obtained:.2%}")
    axes[0, 1].set_xlabel("Hearts Obtained Rate", fontweight="bold")
    axes[0, 1].set_ylabel("Frequency", fontweight="bold")
    axes[0, 1].set_title("Distribution of Hearts Obtained Rates", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Plot 3: Distribution of hearts deposited
    axes[1, 0].hist(df["hearts_deposited"], bins=20, color="orange", alpha=0.7, edgecolor="black")
    mean_deposited = df["hearts_deposited"].mean()
    axes[1, 0].axvline(x=mean_deposited, color="red", linestyle="--", label=f"Mean: {mean_deposited:.2%}")
    axes[1, 0].set_xlabel("Hearts Deposited Rate", fontweight="bold")
    axes[1, 0].set_ylabel("Frequency", fontweight="bold")
    axes[1, 0].set_title("Distribution of Hearts Deposited Rates", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Plot 4: Top missions by gap (obtained - deposited)
    df["gap"] = df["hearts_obtained"] - df["hearts_deposited"]
    top_gaps = df.nlargest(20, "gap")
    axes[1, 1].barh(range(len(top_gaps)), top_gaps["gap"], color="crimson", alpha=0.8, edgecolor="black")
    axes[1, 1].set_yticks(range(len(top_gaps)))
    axes[1, 1].set_yticklabels([f"{row['mission']} ({row['run_id']})" for _, row in top_gaps.iterrows()], fontsize=8)
    axes[1, 1].set_xlabel("Gap (Obtained - Deposited)", fontweight="bold")
    axes[1, 1].set_title("Top 20 Missions by Hearts Gap", fontweight="bold")
    axes[1, 1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hearts_analysis.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved hearts_analysis.png")


def plot_dense_curriculum_comparison(analyses: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot comparison specifically for dense curriculum runs."""
    dense_runs = [a for a in analyses if a.get("run_id", "").startswith("dense_")]

    if not dense_runs:
        print("⚠ No dense curriculum runs found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Dense Curriculum Runs Comparison", fontsize=16, fontweight="bold")

    runs = [a.get("run_id", "unknown") for a in dense_runs]
    mastered = [len(a.get("missions_mastered", [])) for a in dense_runs]
    stuck = [len(a.get("missions_stuck", [])) for a in dense_runs]
    failing = [len(a.get("missions_failing", [])) for a in dense_runs]
    episodes = [a.get("total_episodes", 0) for a in dense_runs]

    x = np.arange(len(runs))
    width = 0.25

    # Plot 1: Mission status
    axes[0, 0].bar(x - width, mastered, width, label="Reward", color="green", alpha=0.8, edgecolor="black")
    axes[0, 0].bar(x, stuck, width, label="Low Heart Deposits", color="orange", alpha=0.8, edgecolor="black")
    axes[0, 0].bar(x + width, failing, width, label="Low Heart Gains", color="red", alpha=0.8, edgecolor="black")
    axes[0, 0].set_xlabel("Run ID", fontweight="bold")
    axes[0, 0].set_ylabel("Number of Missions", fontweight="bold")
    axes[0, 0].set_title("Mission Status", fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Plot 2: Episodes
    axes[0, 1].bar(x, episodes, color="steelblue", alpha=0.8, edgecolor="black")
    axes[0, 1].set_xlabel("Run ID", fontweight="bold")
    axes[0, 1].set_ylabel("Total Episodes", fontweight="bold")
    axes[0, 1].set_title("Training Episodes", fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(episodes):
        axes[0, 1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    # Plot 3: Extract resource levels and map difficulty from run IDs
    resource_levels = []
    map_difficulty = []
    has_diagnostics = []
    for run_id in runs:
        if "rlvl1-10" in run_id:
            resource_levels.append("1-10")
        elif "rlvl7-10" in run_id:
            resource_levels.append("7-10")
        else:
            resource_levels.append("unknown")

        if "easier" in run_id:
            map_difficulty.append("Easier")
        elif "difficult" in run_id:
            map_difficulty.append("Difficult")
        else:
            map_difficulty.append("unknown")

        has_diagnostics.append("diag" in run_id)

    # Group by configuration
    config_groups = {}
    for i, _run_id in enumerate(runs):
        key = f"{resource_levels[i]}_{map_difficulty[i]}_{'diag' if has_diagnostics[i] else 'nodiag'}"
        if key not in config_groups:
            config_groups[key] = {"mastered": [], "stuck": [], "failing": []}
        config_groups[key]["mastered"].append(mastered[i])
        config_groups[key]["stuck"].append(stuck[i])
        config_groups[key]["failing"].append(failing[i])

    configs = sorted(config_groups.keys())
    avg_mastered = [np.mean(config_groups[c]["mastered"]) for c in configs]
    avg_stuck = [np.mean(config_groups[c]["stuck"]) for c in configs]
    avg_failing = [np.mean(config_groups[c]["failing"]) for c in configs]

    x2 = np.arange(len(configs))
    axes[1, 0].bar(x2 - width, avg_mastered, width, label="Reward", color="green", alpha=0.8, edgecolor="black")
    axes[1, 0].bar(x2, avg_stuck, width, label="Low Heart Deposits", color="orange", alpha=0.8, edgecolor="black")
    axes[1, 0].bar(x2 + width, avg_failing, width, label="Low Heart Gains", color="red", alpha=0.8, edgecolor="black")
    axes[1, 0].set_xlabel("Configuration", fontweight="bold")
    axes[1, 0].set_ylabel("Average Number of Missions", fontweight="bold")
    axes[1, 0].set_title("Average Mission Status by Configuration", fontweight="bold")
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Plot 4: Diagnostics vs No Diagnostics
    diag_failing = [failing[i] for i in range(len(runs)) if has_diagnostics[i]]
    nodiag_failing = [failing[i] for i in range(len(runs)) if not has_diagnostics[i]]

    axes[1, 1].boxplot([diag_failing, nodiag_failing], tick_labels=["With Diagnostics", "No Diagnostics"])
    axes[1, 1].set_ylabel("Number of Failing Missions", fontweight="bold")
    axes[1, 1].set_title("Failing Missions: Diagnostics vs No Diagnostics", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "dense_curriculum_comparison.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved dense_curriculum_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from curriculum training analysis JSON files")
    parser.add_argument(
        "--json-dir", type=Path, default=Path("run_analysis/json"), help="Directory containing analysis JSON files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("run_analysis/plots"), help="Output directory for plots"
    )
    parser.add_argument(
        "--wandb-data-dir",
        type=Path,
        default=Path("run_analysis/wandb_data"),
        help="Directory containing cached W&B data",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading analysis files from {args.json_dir}...")
    analyses = load_analysis_files(args.json_dir)
    print(f"Loaded {len(analyses)} analysis files")

    if not analyses:
        print("No analysis files found!", file=sys.stderr)
        sys.exit(1)

    # Load cached W&B data
    wandb_data = load_wandb_data(args.wandb_data_dir)

    print(f"\nGenerating plots in {args.output_dir}/...")
    plot_comparison_summary(analyses, args.output_dir, wandb_data)
    plot_heart_gained_comparison(analyses, args.output_dir, wandb_data)
    plot_per_env_metrics(analyses, args.output_dir, wandb_data)
    plot_average_reward_per_mission(analyses, args.output_dir, wandb_data)
    plot_heart_deposits_per_mission(analyses, args.output_dir, wandb_data)
    plot_heart_gained_per_mission(analyses, args.output_dir, wandb_data)
    plot_mission_heatmap_by_suite(analyses, args.output_dir)
    plot_hearts_obtained_vs_deposited(analyses, args.output_dir)
    plot_dense_curriculum_comparison(analyses, args.output_dir)

    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
