import pandas as pd
import wandb
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from wandb.apis.public.runs import Run


def get_run(
    run_name: str, entity: str = METTA_WANDB_ENTITY, project: str = METTA_WANDB_PROJECT
) -> Run | None:
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error connecting to W&B: {str(e)}")
        print("Make sure you are connected to W&B: `metta status`")
        return None

    try:
        return api.run(f"{entity}/{project}/{run_name}")
    except Exception as e:
        print(f"Error getting run {run_name}: {str(e)}")
        return None


def fetch_metrics(
    run_names: list[str],
    samples: int | None = 1000,
    keys: list[str] | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch metrics from wandb runs.

    Args:
        run_names: List of wandb run names to fetch metrics from
        samples: Number of samples to return. If None, fetches all data points without subsampling.
                Set to a large number (e.g., 10000) for more data without full scan.
                Default is 1000 for backwards compatibility.
        keys: Optional list of specific metric keys to fetch (speeds up fetching)
        min_step: Optional minimum step to fetch from (ONLY works when samples=None)
        max_step: Optional maximum step to fetch to (ONLY works when samples=None)

    Returns:
        Dictionary mapping run names to pandas DataFrames containing the metrics

    Note:
        min_step and max_step parameters ONLY work when samples=None.
        When using sampling (samples is a number), these parameters are ignored because
        wandb's history() method doesn't support step filtering - it returns evenly
        distributed samples across the entire run.

    Warning:
        Using samples=None with large runs can be VERY slow as scan_history()
        loads all data from wandb servers before returning. Consider using
        samples=10000 or samples=50000 for a good balance of data completeness
        and performance.
    """
    metrics_dfs = {}

    for run_name in run_names:
        run = get_run(run_name)
        if run is None:
            continue

        print(
            f"Fetching metrics for {run_name}: {run.state}, {run.created_at}\n{run.url}..."
        )

        try:
            if samples is None:
                # Check if this is likely to be problematic
                last_step = getattr(run, "lastHistoryStep", None)
                if last_step and last_step > 10000000:  # 10 million steps
                    print(
                        f"  ⚠️  ERROR: This run has {last_step:,} steps (over 10 million)!"
                    )
                    print(
                        "     Fetching all data is not practical and will likely timeout."
                    )
                    print(
                        "     SOLUTION: Use samples=50000 for comprehensive sampling instead."
                    )
                    print("     Skipping this run to prevent hanging...")
                elif last_step and last_step > 1000000:  # 1 million steps
                    print(f"  ⚠️  WARNING: This run has {last_step:,} steps!")
                    print("     Fetching all data will take several minutes.")
                    if min_step is None and max_step is None:
                        print("     Proceeding with caution... Press Ctrl+C to cancel.")

                print("  Fetching data using scan_history...")
                if min_step is not None or max_step is not None:
                    print(f"    Step range: {min_step or 0} to {max_step or 'end'}")
                if keys:
                    print(f"    Keys: {keys}")

                # scan_history - this WILL be slow for large datasets
                try:
                    history_records = list(
                        run.scan_history(
                            keys=keys, min_step=min_step, max_step=max_step
                        )
                    )
                    print(f"    Loaded {len(history_records)} records from wandb")
                except Exception as e:
                    print(f"    ERROR: Failed to load data: {str(e)}")
                    print("    Try using samples=10000 instead.")
                    history_records = []

                metrics_df = pd.DataFrame(history_records)
            else:
                # Use sampled history for faster retrieval
                if min_step is not None or max_step is not None:
                    print(
                        f"  ⚠️  Warning: min_step and max_step are ignored when using sampling (samples={samples})"
                    )
                    print(
                        "     To filter by step range, use samples=None for full scan"
                    )

                if keys:
                    print(f"  Fetching {samples} samples for keys: {keys}")
                else:
                    print(f"  Fetching {samples} sampled data points")
                metrics_df: pd.DataFrame = run.history(
                    samples=samples, keys=keys, pandas=True
                )  # type: ignore

            metrics_dfs[run_name] = metrics_df
            print(f"  Fetched {len(metrics_df)} data points.")

            if len(metrics_df) > 0 and "overview/reward" in metrics_df.columns:
                print(
                    f"  Reward: mean={metrics_df['overview/reward'].mean():.4f}, "
                    f"max={metrics_df['overview/reward'].max():.4f}"
                )
            print(f"  Access with `metrics_dfs['{run_name}']`")
            print("")

        except Exception as e:
            print(f"  Error: {str(e)}")
    return metrics_dfs
