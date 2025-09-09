#!/usr/bin/env uv run
"""Fix cost values for existing runs in a sweep by calculating based on runtime."""

import argparse
import logging
from datetime import datetime
from typing import Optional

import wandb
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def get_runtime_hours(run) -> Optional[float]:
    """Get runtime in hours from the run's summary _runtime attribute."""
    try:
        # Runtime is stored as _runtime in run.summary (in seconds)
        if hasattr(run.summary, "get"):
            runtime_seconds = run.summary.get("_runtime", None)
            if runtime_seconds is not None and runtime_seconds > 0:
                runtime_hours = runtime_seconds / 3600.0
                return runtime_hours
        return None
    except Exception as e:
        logger.warning(f"Could not get runtime for run {run.id}: {e}")
        return None


def fix_sweep_costs(
    sweep_id: str,
    entity: str = "metta-research",
    project: str = "metta",
    cost_per_hour: float = 4.6,
    dry_run: bool = False,
) -> None:
    """Fix cost values for all runs in a sweep."""

    console.print(f"[bold cyan]Fixing costs for sweep: {sweep_id}[/bold cyan]")
    console.print(f"Entity: {entity}, Project: {project}")
    console.print(f"Cost per hour: ${cost_per_hour:.2f}")
    console.print(f"Dry run: {dry_run}\n")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch runs for the sweep
    try:
        runs = api.runs(f"{entity}/{project}", filters={"group": sweep_id})
    except Exception as e:
        console.print(f"[red]Error fetching runs: {e}[/red]")
        return

    # Create results table
    table = Table(title=f"Cost Updates for {sweep_id}")
    table.add_column("Run ID", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("Runtime (hrs)", style="green")
    table.add_column("Old Cost", style="red")
    table.add_column("New Cost", style="blue")
    table.add_column("Status", style="magenta")

    updated_count = 0
    skipped_count = 0
    error_count = 0

    for run in runs:
        try:
            # Get runtime from RunInfo
            runtime_hours = get_runtime_hours(run)

            if runtime_hours is None:
                table.add_row(run.id, run.state, "N/A", "N/A", "N/A", "⚠️ No runtime")
                skipped_count += 1
                continue

            # Calculate new cost
            new_cost = cost_per_hour * runtime_hours

            # Get old cost if exists
            old_cost = None
            if hasattr(run.summary, "get"):
                old_cost = run.summary.get("cost", 0.0)
                if old_cost is None:
                    old_cost = 0.0

            # Prepare update
            update_dict = {
                "cost": new_cost,
                "runtime_hours": runtime_hours,
                "cost_per_hour": cost_per_hour,
                "cost_fixed_by_script": True,
                "cost_fixed_at": datetime.now().isoformat(),
            }

            # Also update observation if it exists
            if hasattr(run.summary, "get") and run.summary.get("observation"):
                observation = run.summary.get("observation", {})
                if isinstance(observation, dict):
                    observation["cost"] = new_cost
                    update_dict["observation"] = observation

            # Apply update if not dry run
            if not dry_run:
                run.summary.update(update_dict)
                run.update()
                status = "✅ Updated"
                updated_count += 1
            else:
                status = "🔍 Dry run"

            table.add_row(
                run.id[:20] + "..." if len(run.id) > 20 else run.id,
                run.state,
                f"{runtime_hours:.2f}",
                f"${old_cost:.2f}" if old_cost else "$0.00",
                f"${new_cost:.2f}",
                status,
            )

        except Exception as e:
            logger.error(f"Error processing run {run.id}: {e}")
            table.add_row(
                run.id[:20] + "..." if len(run.id) > 20 else run.id,
                run.state,
                "ERROR",
                "ERROR",
                "ERROR",
                f"❌ {str(e)[:20]}",
            )
            error_count += 1

    # Display results
    console.print(table)

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Updated: {updated_count}")
    console.print(f"  Skipped: {skipped_count}")
    console.print(f"  Errors: {error_count}")

    if dry_run:
        console.print("\n[yellow]This was a dry run. Use --apply to make actual changes.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Fix cost values for runs in a sweep based on runtime")
    parser.add_argument(
        "sweep_id",
        nargs="?",
        default="axel.sweep.test_skypilot_1932",
        help="Sweep ID to fix costs for (default: axel.sweep.test_skypilot_1932)",
    )
    parser.add_argument("--entity", "-e", default="metta-research", help="WandB entity (default: metta-research)")
    parser.add_argument("--project", "-p", default="metta", help="WandB project (default: metta)")
    parser.add_argument("--cost-per-hour", "-c", type=float, default=4.6, help="Cost per hour in USD (default: 4.6)")
    parser.add_argument("--apply", action="store_true", help="Actually apply the changes (default is dry run)")

    args = parser.parse_args()

    # Check WandB authentication
    try:
        import wandb

        if not wandb.api.api_key:
            console.print("[red]Error: WandB API key not found. Please run 'wandb login' first.[/red]")
            return 1
    except Exception as e:
        console.print(f"[red]Error checking WandB authentication: {e}[/red]")
        return 1

    # Run the fix
    try:
        fix_sweep_costs(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            cost_per_hour=args.cost_per_hour,
            dry_run=not args.apply,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
