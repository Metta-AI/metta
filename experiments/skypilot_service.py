"""Service class for interacting with Skypilot."""

import logging
import re
import subprocess
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from metta.common.util.fs import get_repo_root
from metta.common.util.git import (
    get_current_commit,
)
from experiments.training_job import TrainingJob


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
        self._tracked_jobs: Dict[str, TrainingJob] = {}  # job_id -> TrainingJob
        self._jobs_by_name: Dict[str, TrainingJob] = {}  # run_name -> TrainingJob

    def run_preflight_checks(self, git_check: bool = True) -> bool:
        """Run preflight checks using existing utilities.

        Args:
            git_check: Whether to check git state

        Returns:
            True if all checks pass
        """
        all_good = True

        # Git checks (using existing utilities)
        if git_check:
            from devops.skypilot.utils import check_git_state

            commit_hash = get_current_commit()
            error = check_git_state(commit_hash)
            if error:
                print(error)
                all_good = False
            else:
                print("✅ Git state is clean and pushed")

        # AWS check (using existing setup module)
        try:
            from metta.setup.components.aws import AWSSetup

            aws = AWSSetup()
            account = aws.check_connected_as()
            if account:
                print(f"✅ AWS configured (Account: {account})")
            else:
                print("⚠️  AWS credentials may not be configured")
        except Exception as e:
            self.log.debug(f"AWS check failed: {e}")
            print("⚠️  Could not verify AWS credentials")

        # Wandb check (using existing setup module)
        try:
            from metta.setup.components.wandb import WandbSetup

            wandb = WandbSetup()
            if wandb.check_installed():
                print("✅ W&B configured")
            else:
                print("⚠️  W&B not configured (run: wandb login)")
        except Exception as e:
            self.log.debug(f"W&B check failed: {e}")

        return all_good

    def launch_training(
        self,
        run_name: str,
        training_job: TrainingJob,
        instance_name: str,
    ) -> LaunchResult:
        """Launch a training job via Skypilot.

        Args:
            run_name: Name for the training run
            training_job: TrainingJob with full configuration
            instance_name: Instance name for the experiment (includes timestamp)

        Returns:
            LaunchResult with success status and job_id if successful
        """
        if not training_job or not training_job.config:
            raise ValueError("TrainingJob with config is required")

        # Build command with YAML config
        cmd, config_file = self._build_command(
            run_name=run_name,
            training_job=training_job,
            instance_name=instance_name,
        )

        # Execute command
        result = self._execute_command(cmd)

        # Clean up config file if we created one
        if config_file and config_file.exists():
            try:
                config_file.unlink()
            except Exception as e:
                self.log.debug(f"Failed to clean up config file {config_file}: {e}")

        # Track the job if successful
        if result.success and result.job_id:
            # Update the provided TrainingJob
            training_job.job_id = result.job_id
            training_job.launched = True
            training_job.success = result.success
            training_job.launch_time = datetime.now()
            self._tracked_jobs[result.job_id] = training_job
            self._jobs_by_name[run_name] = training_job

        return result

    def _build_command(
        self,
        run_name: str,
        training_job: Optional[TrainingJob],
        instance_name: str,
    ) -> Tuple[List[str], Path]:
        """Build the skypilot launch command from a TrainingJob.

        Args:
            run_name: Name for the run
            training_job: The TrainingJob with full configuration
            instance_name: Instance name for the experiment (includes timestamp)

        Returns:
            Tuple of (command arguments list, config file path)
        """
        if not training_job or not training_job.config:
            raise ValueError("TrainingJob with config is required")

        config = training_job.config

        # Serialize training config to YAML file
        config_file, full_config = config.training.serialize_to_yaml_file(
            instance_name=instance_name
        )

        # Build command with infrastructure settings from skypilot config
        skypilot = config.skypilot
        cmd = [
            "./devops/skypilot/launch.py",
            "train",
            f"run={run_name}",
            f"--gpus={skypilot.gpus}",
            f"--nodes={skypilot.nodes}",
        ]

        # Core Skypilot options
        if not skypilot.spot:
            cmd.append("--no-spot")

        if not skypilot.git_check:
            cmd.append("--skip-git-check")

        # Remove dry_run check - we removed this flag

        # Pass the config file path to launch.py
        cmd.append(f"--config-file={config_file}")

        self.log.info(f"Using serialized config from: {config_file}")
        self.log.debug(
            f"Config contents:\n{yaml.dump(full_config, default_flow_style=False)}"
        )

        return cmd, config_file

    def _execute_command(self, cmd: List[str]) -> LaunchResult:
        """Execute the command and parse results."""
        self.log.info(f"Launching: {' '.join(cmd)}")
        print(f"\n📋 Full command:\n  {' '.join(cmd)}\n")
        print(f"📁 Working directory: {get_repo_root()}\n")

        job_id = None
        success = False
        output_lines = []
        error_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_repo_root(),
            )

            # Read stdout
            for line in process.stdout or []:
                print(line, end="")
                output_lines.append(line.strip())

                # Extract job ID from output
                parsed_id = self._parse_job_id(line)
                if parsed_id:
                    job_id = parsed_id

            # Read stderr
            stderr_output = process.stderr.read() if process.stderr else ""
            if stderr_output:
                error_lines = stderr_output.strip().split("\n")
                print(f"\n⚠️  Stderr output:\n{stderr_output}")

            process.wait()

            # Consider it a success if we got a job ID (even request ID) or return code is 0
            success = process.returncode == 0 or job_id is not None

            if success:
                if job_id and job_id.startswith("request-"):
                    request_id = job_id[8:]  # Remove "request-" prefix
                    self.log.info(
                        f"✓ Job submitted successfully! Request ID: {request_id}"
                    )
                    print(f"\n✅ Job submitted! Request ID: {request_id}")
                    print("   The job is being scheduled. Check status with:")
                    print("   - sky jobs queue")
                    print(f"   - sky api logs {request_id[:8]}")
                elif job_id:
                    self.log.info(f"✓ Job launched successfully! Job ID: {job_id}")
                    print(f"\n✅ Success! Job ID: {job_id}")
                else:
                    self.log.info(
                        "✓ Job appeared to launch (return code 0) but no ID captured"
                    )
                    print(
                        "\n✅ Launch completed (return code 0) - check `sky jobs queue` for status"
                    )
            else:
                self.log.error(
                    f"✗ Launch failed with return code: {process.returncode}"
                )
                print(f"\n❌ Launch failed with return code: {process.returncode}")

                # Print debugging information
                print("\n🔍 Debugging information:")
                print(f"  - Command: {' '.join(cmd)}")
                print(f"  - Working directory: {get_repo_root()}")
                print(f"  - Return code: {process.returncode}")

                if error_lines:
                    print("\n  Error details:")
                    for line in error_lines[-10:]:  # Show last 10 error lines
                        if line.strip():
                            print(f"    {line}")

                # Check for common issues
                if "git" in " ".join(error_lines).lower():
                    print("\n  💡 Hint: This might be a git-related issue. Try:")
                    print("     - Ensure your git repo is clean: `git status`")
                    print("     - Or use --git-check=false to skip git checks")

                if (
                    "authentication" in " ".join(error_lines).lower()
                    or "credentials" in " ".join(error_lines).lower()
                ):
                    print("\n  💡 Hint: This might be an authentication issue. Try:")
                    print("     - Check your cloud credentials: `sky check`")
                    print("     - Ensure you're logged into wandb: `wandb login`")

                if (
                    "resources" in " ".join(error_lines).lower()
                    or "quota" in " ".join(error_lines).lower()
                ):
                    print("\n  💡 Hint: This might be a resource issue. Try:")
                    print("     - Check available resources: `sky show-gpus`")
                    print("     - Use fewer GPUs or nodes")
                    print("     - Use --spot=true for spot instances")

        except FileNotFoundError:
            self.log.error(f"✗ Command not found: {cmd[0]}")
            print(f"\n❌ Error: Command not found: {cmd[0]}")
            print(
                f"  Make sure the launch script exists at: {get_repo_root() / cmd[0]}"
            )
        except Exception as e:
            self.log.error(f"✗ Error launching job: {str(e)}")
            print(f"\n❌ Unexpected error: {str(e)}")
            print(f"  Error type: {type(e).__name__}")

        return LaunchResult(success=success, job_id=job_id)

    def _parse_job_id(self, line: str) -> Optional[str]:
        """Parse job ID from output line."""
        # Look for standard Sky job ID format: sky-YYYY-MM-DD-HH-MM-SS-XXXXXX
        if "sky-" in line:
            parts = line.split()
            for part in parts:
                if part.startswith("sky-") and "-" in part[4:]:
                    return part

        # Also handle API request IDs (when job is submitted but ID not immediately available)
        if "Submitted sky.jobs.launch request:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                request_id = parts[-1].strip()
                # Return a placeholder that indicates submission success
                return f"request-{request_id}"

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
    ) -> pd.DataFrame:
        """Get combined status for training runs.

        Args:
            wandb_run_names: List of wandb run names
            skypilot_job_ids: Optional list of corresponding sky job IDs
            show_metrics: Metrics to include in status
            entity: Wandb entity
            project: Wandb project

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

                # Drop merged columns
                cols_to_drop = ["ID", "STATUS", "JOB DURATION"]
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
