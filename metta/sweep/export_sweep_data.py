#!/usr/bin/env python3
"""Export sweep data to CSV format.

This script pulls all runs from a sweep and converts their observations to a CSV file,
where the columns are the score, cost, and the flattened keys of the suggestion.
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from metta.sweep.stores.wandb import WandbStore
from metta.sweep.models import RunInfo


logger = logging.getLogger(__name__)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested keys
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_observation_data(run: RunInfo) -> Dict[str, Any]:
    """Extract observation data from a run.

    Args:
        run: RunInfo object containing the run data

    Returns:
        Dictionary with score, cost, and flattened suggestion keys
    """
    if not run.observation:
        return {}

    data = {
        'score': run.observation.score,
        'cost': run.observation.cost,
    }

    # Flatten the suggestion dictionary and add to data
    if run.observation.suggestion:
        flattened_suggestion = flatten_dict(run.observation.suggestion)
        data.update(flattened_suggestion)

    return data


def get_all_columns(runs: List[RunInfo]) -> List[str]:
    """Get all unique columns from all runs.

    Args:
        runs: List of RunInfo objects

    Returns:
        List of column names, with score and cost first
    """
    all_keys = set()

    for run in runs:
        if run.observation:
            # Add flattened suggestion keys
            if run.observation.suggestion:
                flattened = flatten_dict(run.observation.suggestion)
                all_keys.update(flattened.keys())

    # Start with score and cost, then add suggestion keys alphabetically
    columns = ['score', 'cost']
    suggestion_keys = sorted([key for key in all_keys if key not in columns])
    columns.extend(suggestion_keys)

    return columns


def export_sweep_to_csv(
    sweep_name: str,
    entity: str = "metta-research",
    project: str = "metta",
    output_dir: str = "data"
) -> str:
    """Export sweep data to CSV file.

    Args:
        sweep_name: Name of the sweep to export
        entity: WandB entity
        project: WandB project
        output_dir: Directory to save the CSV file

    Returns:
        Path to the created CSV file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Initialize WandB store
    store = WandbStore(entity=entity, project=project)

    # Fetch all runs from the sweep
    logger.info(f"Fetching runs for sweep: {sweep_name}")
    runs = store.fetch_runs(filters={"sweep_id": sweep_name})

    if not runs:
        raise ValueError(f"No runs found for sweep: {sweep_name}")

    logger.info(f"Found {len(runs)} runs")

    # Filter runs that have observations (completed runs)
    runs_with_observations = [run for run in runs if run.observation is not None]

    if not runs_with_observations:
        raise ValueError(f"No completed runs with observations found for sweep: {sweep_name}")

    logger.info(f"Found {len(runs_with_observations)} runs with observations")

    # Get all unique columns
    columns = get_all_columns(runs_with_observations)
    logger.info(f"CSV will have {len(columns)} columns: {columns}")

    # Generate filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M")
    filename = f"{sweep_name}_{timestamp}.csv"
    filepath = Path(output_dir) / filename

    # Write CSV file
    logger.info(f"Writing CSV file: {filepath}")

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for run in runs_with_observations:
            observation_data = extract_observation_data(run)

            # Create row with all columns, filling missing values with None
            row = {}
            for col in columns:
                row[col] = observation_data.get(col)

            writer.writerow(row)

    logger.info(f"Successfully exported {len(runs_with_observations)} runs to {filepath}")
    return str(filepath)


def main():
    """CLI entry point for sweep data export."""
    parser = argparse.ArgumentParser(
        description="Export sweep data to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_sweep_data.py my_sweep_name
  python export_sweep_data.py axel.sweep.test_skypilot_1932 --entity metta-research --project metta
  python export_sweep_data.py my_sweep --output-dir results/
        """
    )

    parser.add_argument(
        "sweep_name",
        help="Name of the sweep to export"
    )
    parser.add_argument(
        "--entity", "-e",
        default="metta-research",
        help="WandB entity (default: metta-research)"
    )
    parser.add_argument(
        "--project", "-p",
        default="metta",
        help="WandB project (default: metta)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Output directory for CSV file (default: data)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        filepath = export_sweep_to_csv(
            sweep_name=args.sweep_name,
            entity=args.entity,
            project=args.project,
            output_dir=args.output_dir
        )

        print(f"✅ Successfully exported sweep data to: {filepath}")

    except Exception as e:
        logger.error(f"Failed to export sweep data: {e}")
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
