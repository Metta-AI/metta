#!/usr/bin/env python3
"""Live monitoring script for CVC random maps memory experiments.

Auto-refreshes every 30 seconds to track training progress.

Usage:
    uv run python recipes/experiment/cortex_paper/memory_needed/cvc_random_maps_monitor.py \\
        --run <run_name>

    # One-shot mode (no auto-refresh):
    uv run python recipes/experiment/cortex_paper/memory_needed/cvc_random_maps_monitor.py \\
        --run <run_name> --once

Examples:
    uv run python recipes/experiment/cortex_paper/memory_needed/cvc_random_maps_monitor.py \\
        --run yatharth.memory-needed.xl_memlen128_2b

    uv run python recipes/experiment/cortex_paper/memory_needed/cvc_random_maps_monitor.py \\
        --run yatharth.memory-needed.agsa_2b --once
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Get the directory where this script lives (for saving plots)
SCRIPT_DIR = Path(__file__).parent.resolve()

# WandB settings
ENTITY = "metta-research"
PROJECT = "metta"

# Key metrics to track
METRICS = [
    # Core metrics
    "overview/reward",
    "metric/agent_step",
    "overview/sps",
    "timing_cumulative/sps",
    # Game-specific metrics
    "env_game/assembler.heart.created",
    "env_game/charger.charge.provided",
    "env_agent/heart.gained",
    "env_agent/energy.gained",
    # Training metrics
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "losses/approx_kl",
    # Episode info
    "experience/rewards",
    "_step",
    "_runtime",
]


def find_similar_runs(run_name: str, limit: int = 5) -> list[str]:
    """Find runs with similar names."""
    api = wandb.Api()
    # Extract key parts from the run name to search
    parts = run_name.split(".")
    if len(parts) >= 3:
        # Try searching by user prefix
        search_term = ".".join(parts[:2])
    else:
        search_term = run_name[:20]

    try:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"name": {"$regex": search_term}},
            order="-created_at",
        )
        return [r.name for i, r in enumerate(runs) if i < limit]
    except Exception:
        return []


def fetch_run_data(run_name: str, samples: int = 500) -> tuple[dict, pd.DataFrame | None]:
    """Fetch run summary and history from WandB."""
    api = wandb.Api()

    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_name}")
    except Exception as e:
        print(f"❌ Could not find run: {run_name}")
        print(f"   Error: {e}")
        print("\n🔍 Searching for similar runs...")
        similar = find_similar_runs(run_name)
        if similar:
            print("   Found these similar runs:")
            for name in similar:
                print(f"      • {name}")
        else:
            print("   No similar runs found. The job may still be starting.")
        print("\n💡 Tips:")
        print("   - Check SkyPilot status: sky queue")
        print("   - The job may take a few minutes to start logging")
        print("   - Verify the run name is correct")
        return {}, None

    summary = {
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
    }

    # Add summary metrics
    for key in METRICS:
        if key in run.summary:
            summary[key] = run.summary[key]

    # Fetch history for plotting - don't filter by keys to get all available data
    try:
        history = run.history(samples=samples, pandas=True)
        if history.empty:
            print("⚠️  History is empty")
    except Exception as e:
        print(f"⚠️  Could not fetch history: {e}")
        history = None

    return summary, history


def format_number(n: float | int | None, decimals: int = 2) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n is None:
        return "N/A"
    if abs(n) >= 1e9:
        return f"{n / 1e9:.{decimals}f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def format_metric(value, decimals: int = 4) -> str:
    """Format a metric value, handling None and 0 properly."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return format_number(value, decimals)
    return str(value)


def print_summary(summary: dict) -> None:
    """Print run summary to console."""
    os.system("clear" if os.name != "nt" else "cls")

    print("=" * 70)
    print("🔬 CVC Random Maps Memory Experiment Monitor")
    print("=" * 70)
    print(f"📊 Run: {summary.get('name', 'Unknown')}")
    print(f"🔗 URL: {summary.get('url', 'Unknown')}")
    print(f"📅 Created: {summary.get('created_at', 'Unknown')}")
    print(f"🚦 State: {summary.get('state', 'Unknown')}")
    print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Progress
    steps = summary.get("metric/agent_step") or summary.get("_step")
    target = 2_000_000_000
    if steps is not None:
        pct = (steps / target) * 100
        print(f"\n📈 Progress: {format_number(steps)} / {format_number(target)} ({pct:.2f}%)")
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"   [{bar}]")

        # Runtime info
        runtime = summary.get("_runtime")
        if runtime:
            hours = runtime / 3600
            print(f"   Runtime: {hours:.1f} hours")

    # Performance metrics
    print("\n🎯 Performance Metrics:")
    reward = summary.get("overview/reward")
    print(f"   • Reward:          {format_metric(reward, 4)}")

    hearts = summary.get("env_game/assembler.heart.created")
    print(f"   • Hearts created:  {format_metric(hearts, 4)}")

    charges = summary.get("env_game/charger.charge.provided")
    print(f"   • Charges given:   {format_metric(charges, 4)}")

    hearts_gained = summary.get("env_agent/heart.gained")
    print(f"   • Hearts gained:   {format_metric(hearts_gained, 4)}")

    energy = summary.get("env_agent/energy.gained")
    print(f"   • Energy gained:   {format_metric(energy, 1)}")

    # Training speed
    print("\n⚡ Training Speed:")
    sps = summary.get("overview/sps") or summary.get("timing_cumulative/sps")
    print(f"   • SPS:             {format_metric(sps, 0)}")
    if sps and steps:
        remaining = target - steps
        eta_seconds = remaining / sps
        eta_hours = eta_seconds / 3600
        print(f"   • ETA:             {eta_hours:.1f} hours ({eta_hours / 24:.1f} days)")

    # Loss metrics
    print("\n📉 Loss Metrics:")
    policy_loss = summary.get("losses/policy_loss")
    print(f"   • Policy loss:     {format_metric(policy_loss, 6)}")

    value_loss = summary.get("losses/value_loss")
    print(f"   • Value loss:      {format_metric(value_loss, 6)}")

    entropy = summary.get("losses/entropy")
    print(f"   • Entropy:         {format_metric(entropy, 4)}")

    kl = summary.get("losses/approx_kl")
    print(f"   • Approx KL:       {format_metric(kl, 6)}")

    print("-" * 70)


def plot_metrics(history: pd.DataFrame, run_name: str) -> None:
    """Plot training curves."""
    if history is None or history.empty:
        print("⚠️  No history data to plot")
        return

    # Get step column - try multiple options
    step_col = None
    for col in ["metric/agent_step", "_step"]:
        if col in history.columns:
            step_col = col
            break

    if step_col is None:
        print("⚠️  No step column found in history")
        print(f"   Available columns: {list(history.columns)[:10]}...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"CVC Random Maps: {run_name}\n(Refresh: {datetime.now().strftime('%H:%M:%S')})", fontsize=14, fontweight="bold"
    )

    # Plot 1: Reward over time
    ax = axes[0, 0]
    reward_col = "overview/reward" if "overview/reward" in history.columns else "experience/rewards"
    if reward_col in history.columns:
        df = history[[step_col, reward_col]].dropna()
        if not df.empty and df[reward_col].sum() != 0:
            ax.plot(df[step_col], df[reward_col], color="#2ecc71", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Agent Steps")
            ax.set_ylabel("Reward")
            ax.set_title("Episode Reward")
            ax.grid(True, alpha=0.3)
            if len(df) > 20:
                smoothed = df[reward_col].rolling(window=20, min_periods=1).mean()
                ax.plot(df[step_col], smoothed, color="#27ae60", linewidth=2.5, label="Smoothed")
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Reward = 0\n(no learning yet)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="gray",
            )
            ax.set_title("Episode Reward")
    else:
        ax.text(0.5, 0.5, "No reward data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Episode Reward")

    # Plot 2: Hearts created
    ax = axes[0, 1]
    hearts_col = "env_game/assembler.heart.created"
    if hearts_col in history.columns:
        df = history[[step_col, hearts_col]].dropna()
        if not df.empty and df[hearts_col].sum() != 0:
            ax.plot(df[step_col], df[hearts_col], color="#e74c3c", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Agent Steps")
            ax.set_ylabel("Hearts Created")
            ax.set_title("Assembler Hearts Created (Key Metric)")
            ax.grid(True, alpha=0.3)
            if len(df) > 20:
                smoothed = df[hearts_col].rolling(window=20, min_periods=1).mean()
                ax.plot(df[step_col], smoothed, color="#c0392b", linewidth=2.5, label="Smoothed")
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Hearts = 0\n(no hearts created yet)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="gray",
            )
            ax.set_title("Assembler Hearts Created (Key Metric)")
    else:
        ax.text(0.5, 0.5, "No hearts data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Assembler Hearts Created")

    # Plot 3: SPS over time
    ax = axes[1, 0]
    sps_col = None
    for col in ["overview/sps", "timing_cumulative/sps", "timing_per_epoch/sps"]:
        if col in history.columns:
            sps_col = col
            break

    if sps_col:
        df = history[[step_col, sps_col]].dropna()
        if not df.empty:
            ax.plot(df[step_col], df[sps_col], color="#3498db", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Agent Steps")
            ax.set_ylabel("Steps per Second")
            ax.set_title(f"Training Speed ({sps_col.split('/')[-1]})")
            ax.grid(True, alpha=0.3)
            mean_sps = df[sps_col].mean()
            ax.axhline(y=mean_sps, color="#2980b9", linestyle="--", linewidth=2, label=f"Mean: {mean_sps:,.0f}")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No SPS data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Training Speed (SPS)")

    # Plot 4: Entropy (more informative than losses early in training)
    ax = axes[1, 1]
    if "losses/entropy" in history.columns:
        df = history[[step_col, "losses/entropy"]].dropna()
        if not df.empty:
            ax.plot(df[step_col], df["losses/entropy"], color="#9b59b6", linewidth=1.5, label="Entropy")
            ax.set_xlabel("Agent Steps")
            ax.set_ylabel("Entropy")
            ax.set_title("Policy Entropy (should decrease as policy sharpens)")
            ax.grid(True, alpha=0.3)
            if len(df) > 20:
                smoothed = df["losses/entropy"].rolling(window=20, min_periods=1).mean()
                ax.plot(df[step_col], smoothed, color="#8e44ad", linewidth=2.5, label="Smoothed")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Policy Entropy")

    plt.tight_layout()

    # Save plot with run name in the same directory as this script
    # Sanitize run name for filename (replace dots and slashes)
    safe_run_name = run_name.replace("/", "_").replace(":", "_")
    plot_path = SCRIPT_DIR / f"cvc_random_maps_{safe_run_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {plot_path}")

    # Try to display interactively
    try:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CVC random maps memory experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python %(prog)s --run yatharth.memory-needed.xl_memlen128_2b
  uv run python %(prog)s --run yatharth.memory-needed.agsa_2b --once
        """,
    )
    parser.add_argument("--run", required=True, help="WandB run name to monitor (required)")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no auto-refresh)")
    parser.add_argument("--samples", type=int, default=500, help="Number of history samples to fetch")
    args = parser.parse_args()

    print(f"🔍 Monitoring run: {args.run}")
    print(f"⏱️  Refresh interval: {args.refresh}s")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            summary, history = fetch_run_data(args.run, samples=args.samples)

            if summary:
                print_summary(summary)
                plot_metrics(history, args.run)
            else:
                print("❌ Failed to fetch run data")

            if args.once:
                break

            print(f"\n🔄 Next refresh in {args.refresh} seconds... (Ctrl+C to stop)")
            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
