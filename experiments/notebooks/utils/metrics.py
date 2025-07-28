import pandas as pd
import wandb
from wandb.apis.public.runs import Run


def get_run(
    run_name: str, entity: str = "metta-research", project: str = "metta"
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


def fetch_metrics(run_names: list[str], samples: int = 1000) -> dict[str, pd.DataFrame]:
    metrics_dfs = {}

    for run_name in run_names:
        run = get_run(run_name)
        if run is None:
            continue

        print(
            f"Fetching metrics for {run_name}: {run.state}, {run.created_at}\n{run.url}..."
        )

        try:
            metrics_df: pd.DataFrame = run.history(samples=samples, pandas=True)  # type: ignore
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
