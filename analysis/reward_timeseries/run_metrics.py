"""
CLI to compute and plot reward time-series metrics using metrics_template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .metrics_template import compute_metrics
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from metrics_template import compute_metrics


def plot_metric(series: pd.Series, step: pd.Series, title: str, output: Path) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(step, series)
    ax.set_title(title)
    ax.set_xlabel("agent_step")
    ax.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot LP metrics.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    metrics = compute_metrics(df, window=args.window)

    metrics.to_parquet(args.output_dir / "metrics.parquet", index=False)

    step = metrics["metric/agent_step"]
    plots = {
        "reward_mean": metrics["reward_mean"],
        "reward_std": metrics["reward_std"],
        "reward_slope": metrics["reward_slope"],
        "reward_sps_corr": metrics["reward_sps_corr"],
    }

    for name, series in plots.items():
        plot_metric(series, step, name, args.output_dir / f"{name}.png")

    print(f"Saved metrics + plots to {args.output_dir}")


if __name__ == "__main__":
    main()

