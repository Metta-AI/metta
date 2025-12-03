"""
Prototype helper for LP reward time-series metrics.

Usage:
    python analysis/reward_timeseries/metrics_template.py \
        --data-path analysis/reward_timeseries/data/fixed_maps/seed0/reward_stream.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_metrics(df: pd.DataFrame, window: int = 5000) -> pd.DataFrame:
    """Compute rolling slope, rolling std (noise), and SPS correlation."""
    df = df.sort_values("metric/agent_step").reset_index(drop=True)

    reward_col = "env_agent/heart.amount"
    step_col = "metric/agent_step"
    sps_col = "overview/sps"

    result = pd.DataFrame()
    result[step_col] = df[step_col]

    # Rolling mean and std for reward noise
    result["reward_mean"] = df[reward_col].rolling(window, min_periods=window // 10).mean()
    result["reward_std"] = df[reward_col].rolling(window, min_periods=window // 10).std()

    # Rolling slope using diff of rolling mean
    result["reward_slope"] = result["reward_mean"].diff()

    # SPS rolling mean/std
    result["sps_mean"] = df[sps_col].rolling(window, min_periods=window // 10).mean()
    result["sps_std"] = df[sps_col].rolling(window, min_periods=window // 10).std()

    # Rolling correlation between reward and SPS
    result["reward_sps_corr"] = (
        df[[reward_col, sps_col]]
        .rolling(window, min_periods=window // 10)
        .corr()
        .unstack()
        .loc[:, (reward_col, sps_col)]
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype LP metrics.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    metrics = compute_metrics(df, window=args.window)

    if args.output:
        metrics.to_parquet(args.output, index=False)
        print(f"Saved metrics to {args.output}")
    else:
        print(metrics.head())


if __name__ == "__main__":
    main()


