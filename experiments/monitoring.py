"""Monitoring utilities for training jobs."""

import subprocess
from typing import Dict, List, Optional
import pandas as pd

from metta.common.util.fs import get_repo_root


def get_sky_jobs_data() -> pd.DataFrame:
    """Fetch current sky jobs data.
    
    Returns:
        DataFrame with sky job information
    """
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
            "SUBMITTED": (header_line.find("SUBMITTED"), header_line.find("TOT. DURATION")),
            "TOT. DURATION": (header_line.find("TOT. DURATION"), header_line.find("JOB DURATION")),
            "JOB DURATION": (header_line.find("JOB DURATION"), header_line.find("#RECOVERIES")),
            "#RECOVERIES": (header_line.find("#RECOVERIES"), header_line.find("STATUS")),
            "STATUS": (header_line.find("STATUS"), None),
        }

        # Parse data rows
        data_rows = []
        for line in lines[header_idx + 1 :]:
            if not line.strip() or line.startswith("No ") or line.startswith("Fetching"):
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


def get_training_status(
    wandb_run_ids: List[str],
    skypilot_job_ids: Optional[List[str]] = None,
    show_metrics: Optional[List[str]] = None,
    entity: str = "metta-research",
    project: str = "metta"
) -> pd.DataFrame:
    """Get combined status for training runs.
    
    Args:
        wandb_run_ids: List of wandb run names
        skypilot_job_ids: Optional list of corresponding sky job IDs
        show_metrics: Metrics to include in status
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        DataFrame with combined status information
    """
    # Get wandb status
    from experiments.wandb import get_run_statuses
    wandb_status = get_run_statuses(wandb_run_ids, show_metrics, entity, project)
    
    # If we have sky job IDs, merge with sky status
    if skypilot_job_ids:
        sky_status = get_sky_jobs_data()
        
        # Create mapping of run names to job IDs
        job_mapping = dict(zip(wandb_run_ids, skypilot_job_ids))
        
        # Add sky status to wandb status
        wandb_status["sky_job_id"] = wandb_status["run_name"].map(job_mapping)
        
        if not sky_status.empty:
            # Merge on job ID
            wandb_status = wandb_status.merge(
                sky_status[["ID", "STATUS", "JOB DURATION"]],
                left_on="sky_job_id",
                right_on="ID",
                how="left",
                suffixes=("", "_sky")
            )
            wandb_status["sky_status"] = wandb_status["STATUS"]
            wandb_status["sky_duration"] = wandb_status["JOB DURATION"]
            wandb_status = wandb_status.drop(columns=["ID", "STATUS", "JOB DURATION"])
    
    return wandb_status