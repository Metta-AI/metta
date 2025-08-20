from datetime import datetime

import pandas as pd
import wandb
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def monitor_training_statuses(
    run_names: list[str],
    show_metrics: list[str] | None = None,
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
) -> pd.DataFrame:
    """
    Marimo-compatible version of monitor_training_statuses.
    Returns both the DataFrame and HTML representation for display.
    """
    if show_metrics is None:
        show_metrics = ["_step", "overview/reward"]

    runs = wandb.Api().runs(f"{entity}/{project}", filters={"name": {"$in": run_names}})

    # Collect data for each run
    data = []
    for run_name in run_names:
        run = next((r for r in runs if r.name == run_name), None)
        row = {
            "name": run_name,
            "state": "NOT FOUND",
            "created": None,
            "url": None,
        }

        if run:
            row.update(
                {
                    "name": run_name,
                    "state": run.state,
                    "created": datetime.fromisoformat(run.created_at).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
            )
            if run.summary:
                for metric in show_metrics:
                    if metric in run.summary:
                        value = run.summary[metric]
                        if isinstance(value, float):
                            row[metric] = f"{value:.4f}"
                        else:
                            row[metric] = value
                    else:
                        row[metric] = "-"
            else:
                for metric in show_metrics:
                    row[metric] = "-"
            row["url"] = run.url
        data.append(row)

    return pd.DataFrame(data)
