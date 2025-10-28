from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def fetch_reward_series(
    run_name: str,
    metric_key: str = "overview/reward",
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
    samples: Optional[int] = 2000,
) -> pd.DataFrame:
    """Fetch reward series for a given run from Weights & Biases.

    Args:
        run_name: W&B run name (e.g., "tasha.10.16.shaped_hypers_vit.contrastive.seed1072").
        metric_key: Metric key to retrieve.
        entity: W&B entity (team/user).
        project: W&B project name.
        samples: Number of sampled history points to fetch. If None, retrieves full history.

    Returns:
        DataFrame with columns including metric_key and step information.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_name}")

    if samples is None:
        # Full scan may be slow on large runs
        records = list(run.scan_history(keys=[metric_key]))
        return pd.DataFrame.from_records(records)

    # Sampled history is much faster and sufficient for a quick test
    df: pd.DataFrame = run.history(keys=[metric_key], samples=samples, pandas=True)  # type: ignore
    return df


def plot_reward(df: pd.DataFrame, metric_key: str, output_path: str) -> None:
    """Plot the reward series and save it.

    Args:
        df: DataFrame containing the reward metric.
        metric_key: Column name for the reward metric.
        output_path: File path to save the plot PNG.
    """
    if df.empty or metric_key not in df.columns:
        raise ValueError(f"Metric '{metric_key}' not found or DataFrame is empty")

    plt.figure(figsize=(10, 4))
    # Prefer step column if present; W&B DataFrame often includes '_step'
    x_col = "_step" if "_step" in df.columns else None
    if x_col is None:
        plt.plot(df[metric_key].values, label=metric_key)
        plt.xlabel("index")
    else:
        plt.plot(df[x_col].values, df[metric_key].values, label=metric_key)
        plt.xlabel("step")
    plt.ylabel("reward")
    plt.title(f"{metric_key}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    try:
        plt.show()
    except Exception:
        # Headless environments may not display; the PNG will still be written
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick test: fetch W&B reward series and plot it."
    )
    parser.add_argument(
        "--run",
        default="tasha.10.16.shaped_hypers_vit.contrastive.seed1072",
        help="W&B run name",
    )
    parser.add_argument(
        "--metric",
        default="overview/reward",
        help="Metric key to retrieve",
    )
    parser.add_argument(
        "--entity", default=METTA_WANDB_ENTITY, help="W&B entity (team/user)"
    )
    parser.add_argument(
        "--project", default=METTA_WANDB_PROJECT, help="W&B project name"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of sampled points to fetch (set to 0 for full scan)",
    )
    parser.add_argument(
        "--out",
        default="experiments/wandb_reward_test.png",
        help="Output PNG path for the plot",
    )
    args = parser.parse_args()

    samples_opt: Optional[int] = None if args.samples == 0 else args.samples

    print(
        f"Fetching '{args.metric}' for run '{args.run}' from {args.entity}/{args.project} "
        f"(samples={'full' if samples_opt is None else samples_opt})..."
    )
    df = fetch_reward_series(
        run_name=args.run,
        metric_key=args.metric,
        entity=args.entity,
        project=args.project,
        samples=samples_opt,
    )

    if df.empty:
        raise RuntimeError(
            "No data returned from W&B. Ensure credentials and run name are correct."
        )

    # Quick stats
    if args.metric in df.columns:
        series = df[args.metric].dropna()
        if not series.empty:
            print(
                f"{args.metric}: count={len(series)}, mean={series.mean():.4f}, "
                f"min={series.min():.4f}, max={series.max():.4f}"
            )
    else:
        print(
            f"Metric '{args.metric}' not found in returned columns: {list(df.columns)}"
        )

    plot_reward(df, args.metric, args.out)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
