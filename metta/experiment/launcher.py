"""Experiment launcher - handles job submission and state tracking."""

from datetime import datetime
from pathlib import Path

from metta.experiment.state import ExperimentState, JobState
from metta.jobs.models import JobSpec
from metta.jobs.runner import LocalJob, RemoteJob


class ExperimentLauncher:
    """Handles launching all jobs in an experiment."""

    def __init__(
        self,
        instance_id: str,
        recipe: str,
        jobs: list[JobSpec],
        dry_run: bool = False,
        sequential: bool = False,
    ):
        """Initialize experiment launcher.

        Args:
            instance_id: Unique experiment instance ID
            recipe: Full module path to experiment definition
            jobs: List of jobs to launch
            dry_run: If True, just print what would be launched
            sequential: If True, wait for each job before launching next (v1: not implemented)
        """
        self.instance_id = instance_id
        self.recipe = recipe
        self.jobs = jobs
        self.dry_run = dry_run
        self.sequential = sequential

        self.log_dir = Path("experiments/logs") / instance_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def launch(self) -> int:
        """Launch all jobs.

        Returns:
            0 on success, 1 on failure
        """
        # Validate unique job names
        job_names = [job.name for job in self.jobs]
        if len(job_names) != len(set(job_names)):
            duplicates = [name for name in set(job_names) if job_names.count(name) > 1]
            print(f"Error: Duplicate job names found: {duplicates}")
            return 1

        # Create initial state
        state = ExperimentState(
            experiment_id=self.instance_id,
            recipe=self.recipe,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            updated_at=datetime.utcnow().isoformat(timespec="seconds"),
            status="pending",
            jobs={job.name: JobState(name=job.name, spec=job) for job in self.jobs},
        )

        if self.dry_run:
            print(f"\n[DRY RUN] Would launch experiment: {self.instance_id}")
            print(f"Recipe: {self.recipe}")
            print(f"\nJobs ({len(self.jobs)}):")
            for job in self.jobs:
                print(f"\n  {job.name}:")
                print(f"    Module: {job.module}")
                print(f"    Args: {job.args}")
                print(f"    Overrides: {job.overrides}")
                print(f"    Execution: {job.execution}")
                if job.execution == "remote":
                    print(f"    Resources: {job.gpus} GPU(s), {job.nodes} node(s), spot={job.spot}")
            return 0

        # Save initial state
        state.save()

        print(f"\n{'=' * 80}")
        print(f"Launching Experiment: {self.instance_id}")
        print(f"{'=' * 80}")
        print(f"Recipe: {self.recipe}")
        print(f"Jobs: {len(self.jobs)}")
        print(f"Mode: {'Sequential' if self.sequential else 'Parallel'}")
        print(f"State: experiments/state/{self.instance_id}.json")
        print(f"Logs: experiments/logs/{self.instance_id}/")
        print(f"{'=' * 80}\n")

        # Inject experiment metadata into all jobs
        for job in self.jobs:
            job.metadata["experiment_id"] = self.instance_id
            # Set WandB group for all jobs
            if "group" not in job.args:
                job.args["group"] = self.instance_id

        # Launch jobs
        if self.sequential:
            success = self._launch_sequential(state)
        else:
            success = self._launch_parallel(state)

        if success:
            print(f"\n✅ Successfully launched {len(self.jobs)} job(s)")
            print(f"\nMonitor: ./tools/run.py {self.recipe} mode=monitor")
            return 0
        else:
            print("\n❌ Failed to launch some jobs")
            return 1

    def _launch_parallel(self, state: ExperimentState) -> bool:
        """Launch all jobs in parallel (submit all at once).

        Args:
            state: Experiment state to update

        Returns:
            True if all jobs launched successfully
        """
        all_success = True

        for job_spec in self.jobs:
            print(f"\n{'─' * 80}")
            print(f"Launching: {job_spec.name}")
            print(f"{'─' * 80}")

            success = self._launch_single_job(job_spec, state)
            if not success:
                all_success = False

        return all_success

    def _launch_sequential(self, state: ExperimentState) -> bool:
        """Launch jobs one at a time, waiting for each to complete.

        Note: For v1, this just calls _launch_parallel. Sequential waiting
        can be added in v2 if needed.

        Args:
            state: Experiment state to update

        Returns:
            True if all jobs launched successfully
        """
        return self._launch_parallel(state)

    def _launch_single_job(self, job_spec: JobSpec, state: ExperimentState) -> bool:
        """Launch a single job via RemoteJob or LocalJob based on execution mode.

        Args:
            job_spec: Job specification
            state: Experiment state to update

        Returns:
            True if job launched successfully
        """
        try:
            # Create appropriate job type based on execution mode
            if job_spec.execution == "remote":
                job = RemoteJob(**job_spec.to_remote_job_args(str(self.log_dir)))
            elif job_spec.execution == "local":
                job = LocalJob(**job_spec.to_local_job_args(str(self.log_dir)))
            else:
                raise ValueError(f"Unknown execution mode: {job_spec.execution}")

            # Submit (non-blocking)
            job.submit()

            # Extract job_id (for remote: skypilot job id, for local: process pid)
            if job_spec.execution == "remote":
                if job._job_id:
                    job_id = str(job._job_id)
                else:
                    # Remote job launch failed
                    print("✗ Failed to get job ID (remote launch failed)")
                    state.update_job_status(job_spec.name, status="failed")
                    return False
            else:
                # Local job
                if job._proc:
                    job_id = str(job._proc.pid)
                else:
                    print("✗ Failed to start local process")
                    state.update_job_status(job_spec.name, status="failed")
                    return False

            # Update state
            state.update_job_status(
                job_spec.name,
                status="running",
                job_id=job_id,
                started_at=datetime.utcnow().isoformat(timespec="seconds"),
                logs_path=str(job._get_log_path()),
            )

            mode_str = "remote" if job_spec.execution == "remote" else "local"
            print(f"✓ Launched successfully ({mode_str}, Job ID: {job_id})")
            return True

        except Exception as e:
            print(f"✗ Launch failed: {e}")
            state.update_job_status(job_spec.name, status="failed")
            return False
