import subprocess
from datetime import datetime
from typing import Any

import ipywidgets as widgets
import pandas as pd
import wandb
from IPython.display import display

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.util.fs import get_repo_root


def get_sky_jobs_data() -> pd.DataFrame:
    try:
        result = subprocess.run(
            ["sky", "jobs", "queue"],
            capture_output=True,
            text=True,
            cwd=get_repo_root(),
        )

        if result.returncode != 0:
            print(f"Error running 'sky jobs queue': {result.stderr}")
            return pd.DataFrame()

        lines = result.stdout.strip().split("\n")

        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("ID") and "NAME" in line and "STATUS" in line:
                header_idx = i
                break

        if header_idx is None:
            return pd.DataFrame()

        # Parse using fixed column positions based on the header
        header_line = lines[header_idx]

        # Define column positions based on the header
        col_positions = {
            "ID": (header_line.find("ID"), header_line.find("TASK")),
            "TASK": (header_line.find("TASK"), header_line.find("NAME")),
            "NAME": (header_line.find("NAME"), header_line.find("RESOURCES")),
            "RESOURCES": (header_line.find("RESOURCES"), header_line.find("SUBMITTED")),
            "SUBMITTED": (
                header_line.find("SUBMITTED"),
                header_line.find("TOT. DURATION"),
            ),
            "TOT. DURATION": (
                header_line.find("TOT. DURATION"),
                header_line.find("JOB DURATION"),
            ),
            "JOB DURATION": (
                header_line.find("JOB DURATION"),
                header_line.find("#RECOVERIES"),
            ),
            "#RECOVERIES": (
                header_line.find("#RECOVERIES"),
                header_line.find("STATUS"),
            ),
            "STATUS": (header_line.find("STATUS"), None),
        }

        # Parse data rows
        data_rows = []
        for line in lines[header_idx + 1 :]:
            if (
                not line.strip()
                or line.startswith("No ")
                or line.startswith("Fetching")
            ):
                continue

            row_data = {}
            for col_name, (start, end) in col_positions.items():
                if end is None:
                    value = line[start:].strip()
                else:
                    value = line[start:end].strip()
                row_data[col_name] = value

            if row_data.get("ID"):  # Only add rows with valid ID
                data_rows.append(row_data)

        return pd.DataFrame(data_rows)

    except Exception as e:
        print(f"Error getting sky jobs data: {str(e)}")
        return pd.DataFrame()


def monitor_training_statuses(
    run_names: list[str],
    show_metrics: list[str] | None = None,
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
) -> pd.DataFrame:
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

    df = pd.DataFrame(data)

    if not df.empty:
        display_training_table_widget(df)

    return df


def display_training_table_widget(df: pd.DataFrame) -> None:
    # Create styled HTML table
    html_rows = []

    def wrap_with_component(
        component: str, value: Any, additional_style: str = ""
    ) -> str:
        return f"<{component} style='padding: 8px; text-align: right; {additional_style}'>{value}</{component}>"

    # Header
    header_html = (
        "<tr>"
        + "".join(
            wrap_with_component("th", h, "background-color: #f0f0f0;")
            for h in df.columns
        )
        + "</tr>"
    )

    # Rows with styling
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]

            # Special styling for different columns
            if col == "state":
                color = {
                    "running": "#28a745",
                    "finished": "#007bff",
                    "failed": "#dc3545",
                    "crashed": "#dc3545",
                    "NOT FOUND": "#ffc107",
                }.get(str(value), "#000")
                cell_html = wrap_with_component(
                    "td", value, f"color: {color}; font-weight: bold;"
                )
            elif col == "sky_status":
                color = {
                    "RUNNING": "#28a745",
                    "SUCCEEDED": "#007bff",
                    "FAILED": "#dc3545",
                    "FAILED_CONTROLLER": "#dc3545",
                    "FAILED_PRECHECKS": "#dc3545",
                    "CANCELLED": "#6c757d",
                    "-": "#6c757d",
                }.get(str(value), "#000")
                cell_html = wrap_with_component(
                    "td", value, f"color: {color}; font-weight: bold;"
                )
            elif col == "url" and bool(value):
                cell_html = wrap_with_component(
                    "td",
                    f"<a href='{value}' target='_blank' style='text-decoration: none;'>wandb link</a>",
                )
            else:
                cell_html = wrap_with_component(
                    "td", value if value is not None else "-"
                )
            cells.append(cell_html)

        html_rows.append(
            "<tr style='border-bottom: 1px solid #ddd;'>" + "".join(cells) + "</tr>"
        )

    # Create complete table HTML
    table_html = f"""
    <style>
        .training-table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }}
        .training-table tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    <table class='training-table'>
        <thead>{header_html}</thead>
        <tbody>{"".join(html_rows)}</tbody>
    </table>
    """

    display(widgets.HTML(table_html))
