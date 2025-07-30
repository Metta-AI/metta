"""Service class for interacting with Skypilot."""

import logging
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd

from metta.common.util.fs import get_repo_root


@dataclass
class LaunchResult:
    """Result of a skypilot launch operation."""

    success: bool
    job_id: Optional[str] = None


class SkypilotService:
    """Service for launching and managing Skypilot jobs."""

    def __init__(self):
        self.log = logging.getLogger(__name__)
        # Track all jobs launched through this service
        self._tracked_jobs: Dict[str, "TrainingJob"] = {}  # job_id -> TrainingJob
        self._jobs_by_name: Dict[str, "TrainingJob"] = {}  # run_name -> TrainingJob

    def launch_training(
        self,
        run_name: str,
        curriculum: str,
        gpus: int = 1,
        nodes: int = 1,
        spot: bool = True,
        skip_git_check: bool = False,
        wandb_tags: Optional[List[str]] = None,
        additional_args: Optional[List[str]] = None,
        training_job: Optional["TrainingJob"] = None,
    ) -> LaunchResult:
        """Launch a training job via Skypilot.

        Args:
            run_name: Name for the training run
            curriculum: Path to curriculum config
            gpus: Number of GPUs per node
            nodes: Number of nodes
            spot: Whether to use spot instances
            skip_git_check: Whether to skip git check
            wandb_tags: Tags for wandb
            additional_args: Additional command line arguments
            training_job: Optional TrainingJob object to update

        Returns:
            LaunchResult with success status and job_id if successful
        """
        # Build command
        cmd = self._build_command(
            run_name=run_name,
            curriculum=curriculum,
            gpus=gpus,
            nodes=nodes,
            spot=spot,
            skip_git_check=skip_git_check,
            wandb_tags=wandb_tags,
            additional_args=additional_args,
        )

        # Execute command
        result = self._execute_command(cmd)

        # Track the job if successful
        if result.success and result.job_id:
            if training_job:
                # Update the provided TrainingJob
                training_job.job_id = result.job_id
                training_job.launched = True
                training_job.success = result.success
                training_job.launch_time = datetime.now()
                self._tracked_jobs[result.job_id] = training_job
                self._jobs_by_name[run_name] = training_job
            else:
                # Create a minimal TrainingJob for tracking
                from experiments.training_job import TrainingJob, TrainingJobConfig

                job = TrainingJob(name=run_name)
                job.job_id = result.job_id
                job.launched = True
                job.success = result.success
                job.launch_time = datetime.now()
                job.config = TrainingJobConfig(
                    curriculum=curriculum,
                    gpus=gpus,
                    nodes=nodes,
                    spot=spot,
                    skip_git_check=skip_git_check,
                    wandb_tags=wandb_tags,
                    additional_args=additional_args,
                )
                self._tracked_jobs[result.job_id] = job
                self._jobs_by_name[run_name] = job

        return result

    def _build_command(
        self,
        run_name: str,
        curriculum: str,
        gpus: int,
        nodes: int,
        spot: bool,
        skip_git_check: bool,
        wandb_tags: Optional[List[str]],
        additional_args: Optional[List[str]],
    ) -> List[str]:
        """Build the skypilot launch command."""
        cmd = [
            "./devops/skypilot/launch.py",
            "train",
            f"run={run_name}",
            f"--gpus={gpus}",
            f"--nodes={nodes}",
        ]

        if not spot:
            cmd.append("--no-spot")

        if skip_git_check:
            cmd.append("--skip-git-check")

        cmd.append(f"trainer.curriculum={curriculum}")

        if wandb_tags:
            # Hydra expects list values as comma-separated strings in square brackets
            tags_str = "[" + ",".join(wandb_tags) + "]"
            cmd.append(f"+wandb.tags={tags_str}")

        if additional_args:
            cmd.extend(additional_args)

        return cmd

    def _execute_command(self, cmd: List[str]) -> LaunchResult:
        """Execute the command and parse results."""
        self.log.info(f"Launching: {' '.join(cmd)}")

        job_id = None
        success = False

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=get_repo_root(),
            )

            for line in process.stdout or []:
                print(line, end="")

                # Extract job ID from output
                parsed_id = self._parse_job_id(line)
                if parsed_id:
                    job_id = parsed_id

            process.wait()
            success = process.returncode == 0

            if success:
                self.log.info(f"✓ Job launched successfully! Job ID: {job_id}")
            else:
                self.log.error(
                    f"✗ Launch failed with return code: {process.returncode}"
                )

        except Exception as e:
            self.log.error(f"✗ Error launching job: {str(e)}")

        return LaunchResult(success=success, job_id=job_id)

    def _parse_job_id(self, line: str) -> Optional[str]:
        """Parse job ID from output line."""
        if "Job ID:" in line or "sky-" in line:
            parts = line.split()
            for part in parts:
                if part.startswith("sky-") and "-" in part[4:]:
                    return part
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Sky job.

        Args:
            job_id: Sky job ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["sky", "cancel", "-y", job_id],
                capture_output=True,
                text=True,
                cwd=get_repo_root(),
            )

            success = result.returncode == 0

            if success:
                self.log.info(f"✓ Cancelled job {job_id}")
                # Update tracked job status
                if job_id in self._tracked_jobs:
                    self._tracked_jobs[job_id].cancelled = True
            else:
                self.log.error(
                    f"✗ Failed to cancel job {job_id}: {result.stderr.strip()}"
                )

            return success
        except Exception as e:
            self.log.error(f"✗ Error cancelling job {job_id}: {e}")
            return False

    def get_tracked_jobs(self) -> List["TrainingJob"]:
        """Get all tracked jobs."""
        return list(self._tracked_jobs.values())

    def get_job_by_id(self, job_id: str) -> Optional["TrainingJob"]:
        """Get a tracked job by its Sky job ID."""
        return self._tracked_jobs.get(job_id)

    def get_job_by_name(self, run_name: str) -> Optional["TrainingJob"]:
        """Get a tracked job by its run name."""
        return self._jobs_by_name.get(run_name)

    def add_job(self, job: "TrainingJob") -> None:
        """Add a job to tracking (e.g., for pre-loaded jobs)."""
        if job.job_id:
            self._tracked_jobs[job.job_id] = job
        self._jobs_by_name[job.name] = job

    def get_sky_jobs_data(self, include_all: bool = False) -> pd.DataFrame:
        """Fetch sky jobs data with cost estimates.

        Args:
            include_all: If True, use 'sky jobs queue --all' to include completed jobs

        Returns:
            DataFrame with sky job information including estimated costs
        """
        try:
            cmd = ["sky", "jobs", "queue"]
            if include_all:
                cmd.append("--all")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=get_repo_root(),
            )

            if result.returncode != 0:
                self.log.error(f"Error running 'sky jobs queue': {result.stderr}")
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
                "RESOURCES": (
                    header_line.find("RESOURCES"),
                    header_line.find("SUBMITTED"),
                ),
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

            df = pd.DataFrame(data_rows)

            # Add cost estimation
            if (
                not df.empty
                and "RESOURCES" in df.columns
                and "JOB DURATION" in df.columns
            ):
                df["DURATION_HOURS"] = df["JOB DURATION"].apply(self._parse_duration)

                # Estimate costs - assume spot instances by default
                df["EST_COST_USD"] = df.apply(
                    lambda row: self._estimate_job_cost(
                        row.get("RESOURCES", ""),
                        row.get("DURATION_HOURS", 0),
                        spot=True,
                    ),
                    axis=1,
                )

                # Format cost for display
                df["EST_COST"] = df["EST_COST_USD"].apply(
                    lambda x: f"${x:.2f}" if x > 0 else "-"
                )

            return df

        except Exception as e:
            self.log.error(f"Error getting sky jobs data: {str(e)}")
            return pd.DataFrame()

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to hours."""
        if not duration_str or duration_str == "-":
            return 0.0

        total_hours = 0.0
        # Match patterns like "1d", "2h", "30m", "45s"
        pattern = r"(\d+)([dhms])"
        matches = re.findall(pattern, duration_str.lower())

        for value, unit in matches:
            value = int(value)
            if unit == "d":
                total_hours += value * 24
            elif unit == "h":
                total_hours += value
            elif unit == "m":
                total_hours += value / 60
            elif unit == "s":
                total_hours += value / 3600

        return total_hours

    def _estimate_job_cost(
        self, resources: str, duration_hours: float, spot: bool = True
    ) -> float:
        """Estimate job cost based on resources and duration."""
        if not resources or resources == "-":
            return 0.0

        # Basic GPU pricing (rough estimates in $/hour)
        gpu_prices = {
            "V100": 3.06,
            "A100": 5.12,
            "A100-40GB": 5.12,
            "A100-80GB": 8.00,
            "T4": 0.526,
            "A10G": 1.212,
            "H100": 10.00,
        }

        # Apply spot discount
        spot_discount = 0.5 if spot else 1.0

        # Parse resource string (e.g., "1x A100:8" means 1 node with 8 A100s)
        match = re.match(r"(\d+)x\s*([^:]+):(\d+)", resources)
        if match:
            nodes = int(match.group(1))
            gpu_type = match.group(2).strip()
            gpus_per_node = int(match.group(3))

            # Find matching GPU price
            gpu_price = 0.0
            for gpu_name, price in gpu_prices.items():
                if gpu_name in gpu_type:
                    gpu_price = price
                    break

            # Calculate total cost
            total_gpus = nodes * gpus_per_node
            cost_per_hour = total_gpus * gpu_price * spot_discount
            return cost_per_hour * duration_hours

        return 0.0

    def get_job_status_by_name(self, run_name: str) -> Optional[str]:
        """Get the sky job status for a run by name.

        Args:
            run_name: The wandb run name

        Returns:
            Sky job status string or None if not found
        """
        job = self._jobs_by_name.get(run_name)
        if not job or not job.job_id:
            return None

        sky_df = self.get_sky_jobs_data(include_all=True)
        if sky_df.empty:
            return None

        status_row = sky_df[sky_df["ID"] == job.job_id]
        if not status_row.empty:
            return status_row.iloc[0]["STATUS"]
        return None

    def get_wandb_run_name_from_sky_job(self, sky_job_id: str) -> Optional[str]:
        """Extract wandb run name from a Sky job.

        Args:
            sky_job_id: The Sky job ID (e.g., 'sky-1234-5678')

        Returns:
            The wandb run name if found, None otherwise
        """
        try:
            # First check if we already track this job
            tracked_job = self._tracked_jobs.get(sky_job_id)
            if tracked_job and tracked_job.name:
                return tracked_job.name

            # Otherwise, query Sky for job details
            cmd = ["sky", "jobs", "logs", sky_job_id, "--no-follow"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=get_repo_root(),
            )

            if result.returncode != 0:
                self.log.debug(
                    f"Failed to get logs for job {sky_job_id}: {result.stderr}"
                )
                return None

            # Search for wandb run name pattern in logs
            # Looking for patterns like:
            # - "wandb: Run data is saved locally in ./wandb/run-20240320_123456-abc123"
            # - "wandb: 🚀 View run {run_name} at:"
            # - "run={run_name}"

            lines = result.stdout.split("\n")
            for line in lines:
                # Pattern 1: Direct run= parameter
                if "run=" in line:
                    match = re.search(r"run=([\w\.\-]+)", line)
                    if match:
                        return match.group(1)

                # Pattern 2: wandb run name in logs
                if "wandb:" in line and "View run" in line:
                    match = re.search(r"View run ([\w\.\-]+) at:", line)
                    if match:
                        return match.group(1)

                # Pattern 3: wandb run directory
                if "wandb/run-" in line:
                    # Extract the full run name which often appears nearby
                    # This is less reliable, so we continue searching
                    pass

            return None

        except Exception as e:
            self.log.error(f"Error getting wandb run name from job {sky_job_id}: {e}")
            return None

    def get_training_status(
        self,
        wandb_run_names: List[str],
        skypilot_job_ids: Optional[List[str]] = None,
        show_metrics: Optional[List[str]] = None,
        entity: str = "metta-research",
        project: str = "metta",
        include_costs: bool = True,
    ) -> pd.DataFrame:
        """Get combined status for training runs.

        Args:
            wandb_run_names: List of wandb run names
            skypilot_job_ids: Optional list of corresponding sky job IDs
            show_metrics: Metrics to include in status
            entity: Wandb entity
            project: Wandb project
            include_costs: Whether to include cost information

        Returns:
            DataFrame with combined status information
        """
        # Get wandb status
        from experiments.wandb_service import get_wandb_service

        wandb_service = get_wandb_service()
        wandb_status = wandb_service.get_run_statuses(
            wandb_run_names, show_metrics, entity, project
        )

        # If we have sky job IDs, merge with sky status
        if skypilot_job_ids:
            sky_status = self.get_sky_jobs_data()

            # Create mapping of run names to job IDs
            job_mapping = dict(zip(wandb_run_names, skypilot_job_ids))

            # Add sky status to wandb status
            wandb_status["sky_job_id"] = wandb_status["run_name"].map(job_mapping)

            if not sky_status.empty:
                # Select columns to merge
                merge_cols = ["ID", "STATUS", "JOB DURATION"]
                if include_costs and "EST_COST" in sky_status.columns:
                    merge_cols.append("EST_COST")

                # Merge on job ID
                wandb_status = wandb_status.merge(
                    sky_status[merge_cols],
                    left_on="sky_job_id",
                    right_on="ID",
                    how="left",
                    suffixes=("", "_sky"),
                )
                wandb_status["sky_status"] = wandb_status["STATUS"]
                wandb_status["sky_duration"] = wandb_status["JOB DURATION"]
                if include_costs and "EST_COST" in wandb_status.columns:
                    wandb_status["cost"] = wandb_status["EST_COST"]

                # Drop merged columns
                cols_to_drop = ["ID", "STATUS", "JOB DURATION"]
                if "EST_COST" in wandb_status.columns:
                    cols_to_drop.append("EST_COST")
                wandb_status = wandb_status.drop(columns=cols_to_drop)

        return wandb_status


# Global instance for easy access
_skypilot_service = None


def get_skypilot_service() -> SkypilotService:
    """Get the global SkypilotService instance."""
    global _skypilot_service
    if _skypilot_service is None:
        _skypilot_service = SkypilotService()
    return _skypilot_service
