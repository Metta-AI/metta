"""Simple RunStore HTML table that actually works."""

from IPython.display import HTML


def show_runs(run_store):
    """Display a simple HTML table of runs."""
    runs = run_store.get_all()

    if not runs:
        return HTML("<p>No runs tracked yet.</p>")

    # Build simple HTML table
    html = """
    <style>
        .simple-run-table { 
            border-collapse: collapse; 
            width: 100%; 
            font-family: monospace;
            font-size: 13px;
        }
        .simple-run-table th, .simple-run-table td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }
        .simple-run-table th { 
            background-color: #f2f2f2; 
            font-weight: bold;
        }
        .simple-run-table tr:hover {
            background-color: #f5f5f5;
        }
        .status-running { color: #1976d2; font-weight: bold; }
        .status-pending { color: #f57c00; font-weight: bold; }
        .status-failed { color: #d32f2f; font-weight: bold; }
        .status-completed { color: #388e3c; font-weight: bold; }
    </style>
    
    <table class="simple-run-table">
    <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>SkyPilot</th>
        <th>W&B</th>
        <th>Created</th>
    </tr>
    """

    for run in runs[:50]:  # Show max 50 runs
        status_class = f"status-{run.status.value}"

        sky_info = "-"
        if run.sky:
            sky_info = f"{run.sky.job_id} ({run.sky.status.value})"

        wandb_info = "-"
        if run.wandb:
            wandb_info = f'<a href="{run.wandb.url}" target="_blank">{run.wandb.status.value}</a>'

        created = run.created_at.strftime("%Y-%m-%d %H:%M:%S")

        html += f"""
        <tr>
            <td>{run.run_id}</td>
            <td class="{status_class}">{run.status.value.upper()}</td>
            <td>{sky_info}</td>
            <td>{wandb_info}</td>
            <td>{created}</td>
        </tr>
        """

    html += "</table>"

    return HTML(html)


# Simple functions to use in notebook
def add_run(run_store, run_id):
    """Add a run and show the updated table."""
    run_store.add_run(run_id)
    return show_runs(run_store)


def refresh_run(run_store, run_id):
    """Refresh a run and show the updated table."""
    updated = run_store.refresh_run(run_id)
    print(f"Refreshed {run_id}: Updated={updated}")
    return show_runs(run_store)


def refresh_all(run_store):
    """Refresh all runs and show the updated table."""
    updated = run_store.refresh_all()
    print(f"Refreshed {len(updated)} runs")
    return show_runs(run_store)
