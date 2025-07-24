"""Display utilities for monitoring training status in notebooks."""

from typing import List, Optional, Any
import pandas as pd

# Import data fetching from experiments
from experiments.monitoring import get_training_status


def monitor_training_statuses(
    wandb_run_ids: List[str],
    skypilot_job_ids: Optional[List[str]] = None,
    show_metrics: Optional[List[str]] = None,
    entity: str = "metta-research",
    project: str = "metta",
    return_widget: bool = True,
) -> pd.DataFrame:
    """Monitor training runs with optional widget display.
    
    Args:
        wandb_run_ids: List of wandb run names
        skypilot_job_ids: Optional list of corresponding sky job IDs
        show_metrics: Metrics to display
        entity: Wandb entity
        project: Wandb project
        return_widget: If True, display HTML widget in notebook
        
    Returns:
        DataFrame with status information
    """
    # Get status data
    df = get_training_status(wandb_run_ids, skypilot_job_ids, show_metrics, entity, project)
    
    # Display widget if requested and in notebook environment
    if not df.empty and return_widget:
        try:
            import ipywidgets as widgets
            from IPython.display import display
            html_widget = create_training_table_widget(df)
            if html_widget:
                display(html_widget)
        except ImportError:
            # Not in notebook environment
            pass
    
    return df


def create_training_table_widget(df: pd.DataFrame):
    """Create an HTML widget for displaying training status table.
    
    Args:
        df: DataFrame with training status
        
    Returns:
        HTML widget or None if not in notebook environment
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        return None
    
    # Create styled HTML table
    html_rows = []

    def wrap_with_component(component: str, value: Any, additional_style: str = "") -> str:
        return f"<{component} style='padding: 8px; text-align: right; {additional_style}'>{value}</{component}>"

    # Header
    header_html = (
        "<tr>" + "".join(wrap_with_component("th", h, "background-color: #f0f0f0;") for h in df.columns) + "</tr>"
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
                cell_html = wrap_with_component("td", value, f"color: {color}; font-weight: bold;")
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
                cell_html = wrap_with_component("td", value, f"color: {color}; font-weight: bold;")
            elif col == "url" and bool(value):
                cell_html = wrap_with_component(
                    "td",
                    f"<a href='{value}' target='_blank' style='text-decoration: none;'>wandb link</a>",
                )
            else:
                cell_html = wrap_with_component("td", value if value is not None else "-")
            cells.append(cell_html)

        html_rows.append("<tr style='border-bottom: 1px solid #ddd;'>" + "".join(cells) + "</tr>")

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

    return widgets.HTML(table_html)