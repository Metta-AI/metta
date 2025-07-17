#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.20.1",
#   "rich>=13.0.0",
# ]
# ///

"""
Auto-evaluate navigation policies grouped by curriculum type.
Dynamically discovers curriculum paths instead of using hardcoded mappings.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def get_curriculum_short_name(curriculum_path: str) -> str:
    """
    Generate a short name for a curriculum path.

    Args:
        curriculum_path: Full curriculum path

    Returns:
        Short name for the curriculum
    """
    # Remove common prefixes
    path = curriculum_path.replace("/env/mettagrid/curriculum/", "")
    path = path.replace("/env/mettagrid/arena/", "arena_")
    path = path.replace("env/mettagrid/curriculum/", "")
    path = path.replace("env/mettagrid/arena/", "arena_")

    # Replace slashes with underscores
    path = path.replace("/", "_")

    return path


def get_runs_by_curriculum(
    entity: str = "metta-research",
    project: str = "metta",
    user_filter: Optional[str] = None,
    limit: int = 200,
    state_filter: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Query wandb for runs grouped by curriculum type.

    Args:
        entity: wandb entity
        project: wandb project
        user_filter: optional user filter (e.g., "jacke")
        limit: maximum number of runs to fetch
        state_filter: optional run state filter ("finished", "running", "crashed", etc.)

    Returns:
        Dictionary mapping curriculum short names to lists of run info
    """
    console.print(f"🔍 Querying runs from {entity}/{project}...")

    try:
        api = wandb.Api()

        # Build filters
        filters = {}
        if state_filter:
            filters["state"] = state_filter
        if user_filter:
            filters["display_name"] = {"$regex": f".*{user_filter}.*"}

        # Get runs
        runs = api.runs(f"{entity}/{project}", filters=filters, per_page=limit)

        curriculum_runs = defaultdict(list)
        processed_count = 0

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Processing runs...", total=None)

            for run in runs:
                processed_count += 1
                progress.update(task, description=f"Processing run {processed_count}...")

                # Check if run has curriculum config
                config = run.config
                curriculum_path = None

                if config:
                    # Check various paths where curriculum might be stored
                    if (
                        "trainer" in config
                        and isinstance(config["trainer"], dict)
                        and "curriculum" in config["trainer"]
                    ):
                        curriculum_path = config["trainer"]["curriculum"]
                    elif "curriculum" in config:
                        curriculum_path = config["curriculum"]
                    elif (
                        "train_job" in config
                        and isinstance(config["train_job"], dict)
                        and "curriculum" in config["train_job"]
                    ):
                        curriculum_path = config["train_job"]["curriculum"]

                if curriculum_path:
                    curriculum_short_name = get_curriculum_short_name(curriculum_path)

                    run_info = {
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "created_at": run.created_at,
                        "curriculum_path": curriculum_path,
                        "curriculum_short_name": curriculum_short_name,
                        "user": run.user.name if run.user else "unknown",
                        "summary": dict(run.summary) if run.summary else {},
                        "tags": run.tags if run.tags else [],
                    }
                    curriculum_runs[curriculum_short_name].append(run_info)

    except Exception as e:
        console.print(f"❌ Error querying wandb: {e}")
        return {}

    # Convert defaultdict to regular dict and sort runs by creation time within each curriculum
    result = {}
    for curriculum_name, runs in curriculum_runs.items():
        result[curriculum_name] = sorted(runs, key=lambda x: x["created_at"], reverse=True)

    return result


def display_runs_summary(curriculum_runs: Dict[str, List[Dict]]) -> None:
    """Display a summary table of found runs."""
    if not curriculum_runs:
        console.print("❌ No runs found with curriculum configurations.")
        return

    table = Table(title="Found Runs by Curriculum")
    table.add_column("Curriculum", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Latest Run", style="green")
    table.add_column("User", style="yellow")
    table.add_column("Full Path", style="dim")

    for curriculum_name, runs in sorted(curriculum_runs.items()):
        if runs:
            latest = runs[0]  # Already sorted by creation time
            table.add_row(curriculum_name, str(len(runs)), latest["name"], latest["user"], latest["curriculum_path"])

    console.print(table)


def run_evaluation(run_info: Dict, curriculum_name: str, dry_run: bool = False) -> bool:
    """
    Run navigation evaluation for a specific run.

    Args:
        run_info: Dictionary containing run information
        curriculum_name: curriculum short name for organizing results
        dry_run: if True, only print the command instead of executing

    Returns:
        True if evaluation succeeded, False otherwise
    """
    run_name = run_info["name"]
    eval_run_name = f"{curriculum_name}_eval_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        "./tools/sim.py",
        "sim=navigation",
        f"run={eval_run_name}",
        f"policy_uri=wandb://run/{run_name}",
        "sim_job.stats_db_uri=wandb://stats/navigation_db",
        "device=cpu",
    ]

    if dry_run:
        console.print(f"🔍 Would run: {' '.join(cmd)}")
        return True

    console.print(f"🚀 Running evaluation for {run_name} ({curriculum_name})")
    console.print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.returncode == 0:
            console.print(f"✅ Evaluation completed for {run_name}")
            return True
        else:
            console.print(f"❌ Evaluation failed for {run_name}")
            console.print(f"   Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        console.print(f"⏰ Evaluation timed out for {run_name}")
        return False
    except Exception as e:
        console.print(f"❌ Error running evaluation for {run_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto-evaluate navigation policies grouped by curriculum type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all runs by curriculum
  %(prog)s --list-only

  # List runs for specific user
  %(prog)s --list-only --user jacke

  # Run evaluations for all found runs
  %(prog)s --user jacke

  # Run evaluations for specific curriculum only
  %(prog)s --user jacke --curriculum simple

  # Dry run (show what would be executed)
  %(prog)s --user jacke --dry-run

  # See available curricula
  %(prog)s --user jacke --show-curricula
        """,
    )

    parser.add_argument("--user", help="Filter runs by username (e.g., jacke)")
    parser.add_argument("--curriculum", help="Only evaluate runs with this curriculum type (use short name)")
    parser.add_argument("--entity", default="metta-research", help="wandb entity")
    parser.add_argument("--project", default="metta", help="wandb project")
    parser.add_argument("--limit", type=int, default=200, help="Maximum runs to fetch")
    parser.add_argument("--state", help="Filter by run state (finished, crashed, failed, etc.)")
    parser.add_argument("--list-only", action="store_true", help="Only list runs, don't run evaluations")
    parser.add_argument("--dry-run", action="store_true", help="Show commands that would be run")
    parser.add_argument("--max-runs-per-curriculum", type=int, default=3, help="Max runs to evaluate per curriculum")
    parser.add_argument("--show-curricula", action="store_true", help="Show available curricula and exit")

    args = parser.parse_args()

    # Query runs
    curriculum_runs = get_runs_by_curriculum(
        entity=args.entity, project=args.project, user_filter=args.user, limit=args.limit, state_filter=args.state
    )

    # Show curricula if requested
    if args.show_curricula:
        if curriculum_runs:
            console.print("Available curricula:")
            for curriculum_name in sorted(curriculum_runs.keys()):
                console.print(f"  {curriculum_name}")
        else:
            console.print("No curricula found.")
        return

    # Display summary
    display_runs_summary(curriculum_runs)

    if args.list_only:
        # Save run info to JSON for reference
        output_file = f"curriculum_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(curriculum_runs, f, indent=2, default=str)
        console.print(f"📝 Run details saved to {output_file}")
        return

    # Filter by curriculum if specified
    if args.curriculum:
        if args.curriculum not in curriculum_runs:
            console.print(f"❌ No runs found for curriculum: {args.curriculum}")
            available = ", ".join(sorted(curriculum_runs.keys()))
            console.print(f"Available curricula: {available}")
            return
        curriculum_runs = {args.curriculum: curriculum_runs[args.curriculum]}

    # Run evaluations
    console.print("\n🎯 Starting evaluations...")

    success_count = 0
    total_count = 0

    for curriculum_name, runs in curriculum_runs.items():
        console.print(f"\n📚 Processing curriculum: {curriculum_name}")

        # Limit runs per curriculum
        runs_to_process = runs[: args.max_runs_per_curriculum]

        for run_info in runs_to_process:
            total_count += 1
            if run_evaluation(run_info, curriculum_name, args.dry_run):
                success_count += 1

    # Summary
    if not args.dry_run:
        console.print(f"\n📊 Evaluation Summary: {success_count}/{total_count} succeeded")
        if success_count > 0:
            console.print("\n🎉 Next steps:")
            console.print("1. View results in the navigation dashboard:")
            console.print(
                "   ./tools/dashboard.py +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html"
            )
            console.print("2. Or analyze specific results:")
            console.print("   ./tools/analyze.py +eval_db_uri=wandb://stats/navigation_db run=analysis")


if __name__ == "__main__":
    main()
