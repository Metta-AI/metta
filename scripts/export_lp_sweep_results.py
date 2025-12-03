"""Export LP sweep results from WandB to CSV for analysis.

Usage:
    uv run python scripts/export_lp_sweep_results.py --group lp_local_grid
    uv run python scripts/export_lp_sweep_results.py --group lp_local_grid --output my_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import wandb


def export_sweep_results(group_name: str, output_file: str = "lp_sweep_results.csv") -> None:
    """Pull sweep results from WandB and export to CSV."""
    api = wandb.Api()

    # Query runs in the specified group
    runs = api.runs(
        path="metta-research/metta",
        filters={"group": group_name},
        order="+created_at",
    )

    results = []
    for run in runs:
        try:
            # Extract hyperparameters from config
            config = run.config

            # Extract key metrics from summary (final values)
            summary = run.summary

            result = {
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "created_at": run.created_at,
                "duration_seconds": (run.summary.get("_runtime", 0)),

                # Hyperparameters
                "ema_timescale": config.get("ema_timescale", None),
                "progress_smoothing": config.get("progress_smoothing", None),
                "exploration_bonus": config.get("exploration_bonus", None),
                "num_cogs": config.get("num_cogs", None),
                "total_timesteps": config.get("trainer", {}).get("total_timesteps", None),

                # Key metrics
                "final_reward": summary.get("experience/rewards", None),
                "heart_gained": summary.get("env_agent/heart.gained", None),
                "steps_per_second": summary.get("perf/steps_per_second", None),

                # Evaluation metrics (if available)
                "eval_score": summary.get("evaluator/eval_cogs_vs_clips/score", None),

                # WandB link
                "url": run.url,
            }

            results.append(result)
            print(f"✓ Exported {run.name} ({run.state})")

        except Exception as e:
            print(f"⚠ Error processing run {run.name}: {e}")
            continue

    if not results:
        print(f"❌ No runs found in group '{group_name}'")
        return

    # Write to CSV
    output_path = Path(output_file)
    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Exported {len(results)} runs to {output_path.absolute()}")
    print(f"\nTop hyperparameters by heart_gained:")

    # Sort and display top results
    sorted_results = sorted(
        [r for r in results if r["heart_gained"] is not None],
        key=lambda x: x["heart_gained"],
        reverse=True,
    )

    for i, result in enumerate(sorted_results[:5], 1):
        print(
            f"  {i}. {result['run_name']}: "
            f"heart_gained={result['heart_gained']:.3f}, "
            f"ema={result['ema_timescale']}, "
            f"smoothing={result['progress_smoothing']}, "
            f"exploration={result['exploration_bonus']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LP sweep results to CSV")
    parser.add_argument("--group", required=True, help="WandB run group name")
    parser.add_argument("--output", default="lp_sweep_results.csv", help="Output CSV file")
    args = parser.parse_args()

    export_sweep_results(args.group, args.output)


if __name__ == "__main__":
    main()

