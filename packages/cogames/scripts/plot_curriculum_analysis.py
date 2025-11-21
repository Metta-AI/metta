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


def _get_wandb_metrics(run_id: str, project: str = "metta") -> dict[str, float]:
    """Fetch actual metrics from W&B for a run."""
    try:
        import wandb
    except ImportError:
        return {"reward": 0.0, "heart_deposits": 0.0, "heart_gains": 0.0}

    try:
        api = wandb.Api(timeout=120)
        run = api.run(f"{project}/{run_id}")
        history = run.history()

        # Get average reward (try overview/reward first, then sum of per-label rewards)
        reward = 0.0
        if "overview/reward" in history.columns:
            reward = history["overview/reward"].mean()
        else:
            # Sum average rewards across all missions
            reward_cols = [col for col in history.columns if col.startswith("env_per_label_rewards/") and ".avg" in col]
            if reward_cols:
                reward = history[reward_cols].sum(axis=1).mean()

        # Get heart deposits rate
        heart_deposits = 0.0
        if "env_game/chest.heart.deposited" in history.columns:
            heart_deposits = (history["env_game/chest.heart.deposited"] > 0).mean()

        # Get heart gains rate
        heart_gains = 0.0
        if "env_agent/heart.gained" in history.columns:
            heart_gains = (history["env_agent/heart.gained"] > 0).mean()

        return {"reward": reward, "heart_deposits": heart_deposits, "heart_gains": heart_gains}
    except Exception as e:
        print(f"Warning: Failed to fetch W&B metrics for {run_id}: {e}", file=sys.stderr)
        return {"reward": 0.0, "heart_deposits": 0.0, "heart_gains": 0.0}


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


def plot_comparison_summary(analyses: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot comparison of actual metrics (heart deposits, heart gains) across runs."""
    runs = []
    heart_deposits = []
    heart_gains = []
    episodes = []
    agent_steps = []

    print("Fetching W&B metrics for all runs...")
    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        runs.append(run_id)
        metrics = _get_wandb_metrics(run_id)
        heart_deposits.append(metrics["heart_deposits"])
        heart_gains.append(metrics["heart_gains"])
        episodes.append(analysis.get("total_episodes", 0))
        agent_steps.append(analysis.get("agent_steps", 0) / 1e9)  # Convert to billions

    x = np.arange(len(runs))
    width = 0.35

    # Plot 1: Actual metrics (heart deposits, heart gains)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, heart_deposits, width, label="Heart Deposits", color="orange", alpha=0.8, edgecolor="black")
    ax.bar(x + width / 2, heart_gains, width, label="Heart Gains", color="red", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Run ID", fontweight="bold")
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("Heart Deposits and Heart Gains by Run", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_mission_status.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_mission_status.png")

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

    # Plot 4: Stacked metrics (normalized to show proportions)
    fig, ax = plt.subplots(figsize=(14, 7))
    # Normalize metrics to percentages for stacked view
    if heart_deposits or heart_gains:
        max_val = max(max(heart_deposits), max(heart_gains))
    else:
        max_val = 1.0
    if max_val > 0:
        normalized_deposits = [d / max_val * 100 for d in heart_deposits]
        normalized_gains = [g / max_val * 100 for g in heart_gains]
    else:
        normalized_deposits = heart_deposits
        normalized_gains = heart_gains

    ax.bar(x, normalized_deposits, width, label="Heart Deposits", color="orange", alpha=0.8, edgecolor="black")
    ax.bar(
        x,
        normalized_gains,
        width,
        bottom=normalized_deposits,
        label="Heart Gains",
        color="red",
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_xlabel("Run ID", fontweight="bold")
    ax.set_ylabel("Normalized Metric Value (%)", fontweight="bold")
    ax.set_title("Stacked Metrics (Normalized)", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_stacked_status.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved comparison_stacked_status.png")


def plot_average_reward_per_mission(analyses: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot average reward per mission across all runs."""
    # Collect all missions and their rewards
    mission_rewards: dict[str, list[float]] = defaultdict(list)
    mission_runs: dict[str, list[str]] = defaultdict(list)

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")

        # Get rewards from stuck missions (they have avg_reward)
        for stuck in analysis.get("missions_stuck", []):
            if isinstance(stuck, dict):
                mission = stuck.get("mission", "")
                reward = stuck.get("avg_reward", 0.0)
                if mission and reward > 0:
                    mission_rewards[mission].append(reward)
                    mission_runs[mission].append(run_id)

        # Get rewards from mastered missions (need to fetch from W&B or use default)
        for mission in analysis.get("missions_mastered", []):
            # Try to get reward from W&B
            metrics = _get_wandb_metrics(run_id)
            if metrics["reward"] > 0:
                mission_rewards[mission].append(metrics["reward"])
                mission_runs[mission].append(run_id)

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
    for i, (bar, reward) in enumerate(zip(bars, avg_rewards)):
        ax.text(reward, i, f"{reward:.2f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "average_reward_per_mission.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved average_reward_per_mission.png")


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
                    (r, m, d)
                    for (r, m), d in mission_data.items()
                    if r == run_id and d["base"] == mission_base
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

        fig, ax = plt.subplots(
            figsize=(max(12, len(missions_list) * 0.5), max(6, len(runs) * 0.3))
        )
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
        title = f"Mission Performance Heatmap - {suite}\n(Green=Reward, Orange=Low Heart Deposits, Red=Low Heart Gains, Gray=Not Present)"
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading analysis files from {args.json_dir}...")
    analyses = load_analysis_files(args.json_dir)
    print(f"Loaded {len(analyses)} analysis files")

    if not analyses:
        print("No analysis files found!", file=sys.stderr)
        sys.exit(1)

    print(f"\nGenerating plots in {args.output_dir}/...")
    plot_comparison_summary(analyses, args.output_dir)
    plot_average_reward_per_mission(analyses, args.output_dir)
    plot_mission_heatmap_by_suite(analyses, args.output_dir)
    plot_hearts_obtained_vs_deposited(analyses, args.output_dir)
    plot_dense_curriculum_comparison(analyses, args.output_dir)

    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
