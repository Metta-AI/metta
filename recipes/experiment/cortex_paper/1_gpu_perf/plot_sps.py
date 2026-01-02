#!/usr/bin/env python3
"""
Fetch and plot SPS metrics for GPU scaling runs.
"""

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# WandB settings
ENTITY = "metta-research"
PROJECT = "metta"

# Run names from reference_code.md (lines 113-116) - 2b series
RUN_NAMES = [
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xL4.2b",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.4xL4.2b",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.8xL4.2b",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xH100.2b",
]

# Labels for the plot
LABELS = {
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xL4.2b": "1x L4",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.4xL4.2b": "4x L4",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.8xL4.2b": "8x L4",
    "yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xH100.2b": "1x H100",
}

# WandB metric keys to try for SPS (in order of preference)
SPS_METRIC_KEYS = [
    "overview/sps",
    "overview/steps_per_second",
    "sps",
    "samples_per_second",
    "training/sps",
    "train/sps",
    "metric/samples_per_second",
    "perf/samples_per_second",
]


def get_sps_key(run) -> str | None:
    """Find which SPS key exists in the run."""
    # Check in summary
    for key in SPS_METRIC_KEYS:
        if key in run.summary and run.summary[key] is not None:
            return key

    # Check in history
    try:
        history = run.history(samples=1)
        if history is not None and len(history) > 0:
            for key in SPS_METRIC_KEYS:
                if key in history.columns:
                    return key
    except Exception:
        pass

    return None


def fetch_sps_history(run, sps_key: str, samples: int = 500) -> pd.DataFrame:
    """Fetch SPS history for a run."""
    step_key = "metric/agent_step"
    keys = [sps_key, step_key, "_runtime"]

    history = run.history(samples=samples, keys=keys, pandas=True)
    return history


def main():
    api = wandb.Api()

    # Fetch data for each run
    run_data = {}

    for run_name in RUN_NAMES:
        print(f"Fetching: {run_name}")
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_name}")
            print(f"  State: {run.state}")

            # Find SPS key
            sps_key = get_sps_key(run)
            if sps_key is None:
                print("  No SPS metric found!")
                continue

            print(f"  SPS key: {sps_key}")

            # Fetch history
            history = fetch_sps_history(run, sps_key)

            if history.empty:
                print("  No history data!")
                continue

            # Get step key
            step_key = "metric/agent_step" if "metric/agent_step" in history.columns else "_step"

            # Clean data - include runtime if available
            cols_to_use = [step_key, sps_key]
            if "_runtime" in history.columns:
                cols_to_use.append("_runtime")

            df = history[cols_to_use].dropna()
            rename_map = {step_key: "step", sps_key: "sps"}
            if "_runtime" in df.columns:
                rename_map["_runtime"] = "runtime"
            df = df.rename(columns=rename_map)
            df = df.sort_values("step")

            run_data[run_name] = df
            print(f"  Fetched {len(df)} data points")
            print(f"  SPS range: {df['sps'].min():,.0f} - {df['sps'].max():,.0f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not run_data:
        print("No data fetched!")
        return

    # Create 2x2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color palette
    colors = {
        "1x L4": "#1f77b4",
        "4x L4": "#ff7f0e",
        "8x L4": "#2ca02c",
        "1x H100": "#d62728",
    }

    # Calculate max SPS for consistent y-axis
    all_sps = [df["sps"].max() for df in run_data.values()]
    max_sps = max(all_sps) if all_sps else 300000
    y_max = max_sps * 1.3

    # === Top Left: SPS vs Agent Steps ===
    ax1 = axes[0, 0]
    for run_name, df in run_data.items():
        label = LABELS[run_name]
        ax1.plot(
            df["step"] / 1e6,
            df["sps"],
            label=label,
            color=colors.get(label),
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=8,
        )

    ax1.set_xlabel("Agent Steps (millions)", fontsize=12)
    ax1.set_ylabel("Samples Per Second (SPS)", fontsize=12)
    ax1.set_title("SPS vs Agent Steps", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, y_max)

    # === Top Right: Average SPS bar chart ===
    ax2 = axes[0, 1]
    avg_sps = []
    bar_colors = []
    bar_labels = []

    for run_name in RUN_NAMES:
        if run_name in run_data:
            label = LABELS[run_name]
            bar_labels.append(label)
            avg_sps.append(run_data[run_name]["sps"].mean())
            bar_colors.append(colors.get(label))

    bars = ax2.barh(bar_labels, avg_sps, color=bar_colors)

    # Add value labels
    for bar, val in zip(bars, avg_sps, strict=True):
        ax2.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2, f"{val:,.0f}", va="center", fontsize=10)

    ax2.set_xlabel("Average SPS", fontsize=12)
    ax2.set_title("Average SPS by Configuration", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # === Bottom Left: SPS vs Wall Clock Time ===
    ax3 = axes[1, 0]
    for run_name, df in run_data.items():
        label = LABELS[run_name]
        if "runtime" in df.columns:
            # Convert runtime from seconds to minutes
            ax3.plot(
                df["runtime"] / 60,
                df["sps"],
                label=label,
                color=colors.get(label),
                linewidth=2,
                alpha=0.8,
                marker="o",
                markersize=8,
            )

    ax3.set_xlabel("Wall Clock Time (minutes)", fontsize=12)
    ax3.set_ylabel("Samples Per Second (SPS)", fontsize=12)
    ax3.set_title("SPS vs Wall Clock Time", fontsize=14, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, y_max)

    # === Bottom Right: Steps achieved vs Wall Clock Time ===
    ax4 = axes[1, 1]
    for run_name, df in run_data.items():
        label = LABELS[run_name]
        if "runtime" in df.columns:
            ax4.plot(
                df["runtime"] / 60,
                df["step"] / 1e6,
                label=label,
                color=colors.get(label),
                linewidth=2,
                alpha=0.8,
                marker="o",
                markersize=8,
            )

    ax4.set_xlabel("Wall Clock Time (minutes)", fontsize=12)
    ax4.set_ylabel("Agent Steps (millions)", fontsize=12)
    ax4.set_title("Training Progress vs Wall Clock Time", fontsize=14, fontweight="bold")
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure to same directory as this script
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "sps_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    # Also show
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for run_name in RUN_NAMES:
        if run_name in run_data:
            label = LABELS[run_name]
            df = run_data[run_name]
            print(f"{label:10s}: avg={df['sps'].mean():>10,.0f} SPS, max={df['sps'].max():>10,.0f} SPS")


if __name__ == "__main__":
    main()
