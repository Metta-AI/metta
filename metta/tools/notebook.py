"""Tool for launching parallel training jobs and generating analysis notebooks."""

import time
import uuid
from pathlib import Path

from pydantic import Field, model_validator

from metta.common.tool import Tool
from metta.common.util.fs import get_repo_root
from metta.common.util.log_config import getRankAwareLogger
from metta.jobs.job_config import JobConfig, RemoteConfig
from metta.jobs.job_display import JobDisplay
from metta.jobs.job_manager import JobManager
from metta.jobs.notebook_generation import generate_experiment_notebook

logger = getRankAwareLogger(__name__)


class NotebookTool(Tool):
    """Launch parallel training jobs and optionally generate analysis notebooks.

    This tool uses the metta/jobs system to launch multiple training jobs in parallel,
    monitors their progress, and can generate a Jupyter notebook with reward and SPS
    graphs for all runs.

    Example usage in a recipe:
        def notebook(runs: list[str]) -> NotebookTool:
            return NotebookTool(
                module="arena.train",
                runs=runs,
                args_per_run={
                    "baseline": ["trainer.total_timesteps=1000000"],
                    "high_lr": ["trainer.total_timesteps=1000000", "trainer.learning_rate=0.001"],
                },
                gpus=4,
                generate_notebook=True,
            )
    """

    module: str = Field(description="Tool module to run (e.g., 'arena.train')")
    runs: list[str] = Field(description="List of run names for the experiment")
    args_per_run: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Additional arguments for each run (keyed by run name)",
    )

    group: str | None = Field(
        default=None,
        description="Job group name. If None, a unique group ID will be generated.",
    )
    gpus: int = Field(default=1, description="Number of GPUs per job")
    timeout_s: int = Field(default=7200, description="Timeout per job in seconds")
    remote: bool = Field(default=True, description="Run jobs remotely via SkyPilot")
    is_training_job: bool = Field(default=True, description="Mark jobs as training jobs")
    metrics_to_track: list[str] = Field(
        default_factory=lambda: ["overview/sps", "overview/reward"],
        description="WandB metrics to track",
    )

    generate_notebook: bool = Field(
        default=True,
        description="Generate a Jupyter notebook with analysis after jobs complete",
    )
    notebook_path: Path | None = Field(
        default=None,
        description="Path to save the generated notebook. If None, uses experiments/notebooks/<group>.ipynb",
    )

    state_dir: Path = Field(
        default_factory=lambda: get_repo_root() / "job_state",
        description="Directory for job state persistence",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "NotebookTool":
        if not self.runs:
            raise ValueError("Must specify at least one run")

        if self.args_per_run:
            # Validate that all keys in args_per_run are in runs
            unknown_runs = set(self.args_per_run.keys()) - set(self.runs)
            if unknown_runs:
                raise ValueError(f"Unknown runs in args_per_run: {unknown_runs}")

        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        # Generate unique group ID if not provided
        if self.group is None:
            self.group = f"experiment_{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting experiment with group: {self.group}")
        logger.info(f"Running {len(self.runs)} jobs: {', '.join(self.runs)}")

        # Initialize job manager
        manager = JobManager(base_dir=self.state_dir)

        # Submit all jobs
        for run_name in self.runs:
            job_args = [f"run={run_name}"]
            # Add any run-specific arguments
            if run_name in self.args_per_run:
                job_args.extend(self.args_per_run[run_name])

            # Create job config
            job_config = JobConfig(
                name=run_name,
                module=self.module,
                args=job_args,
                timeout_s=self.timeout_s,
                remote=RemoteConfig(gpus=self.gpus) if self.remote else None,
                is_training_job=self.is_training_job,
                metrics_to_track=self.metrics_to_track,
                group=self.group,
            )

            logger.info(f"Submitting job: {run_name}")
            manager.submit(job_config)

        # Monitor jobs
        display = JobDisplay(manager, group=self.group)
        logger.info("\nMonitoring jobs...")

        all_complete = False
        while not all_complete:
            manager.poll()
            display.display_status(clear_screen=False, show_running_logs=True)

            # Check if all jobs in this group are complete
            group_jobs = manager.get_group_jobs(self.group)
            all_complete = all(manager.get_job_state(job_name).is_terminal() for job_name in group_jobs)

            if not all_complete:
                time.sleep(2)

        logger.info("\nAll jobs complete!")

        # Check for failures
        failed_jobs = []
        for job_name in manager.get_group_jobs(self.group):
            state = manager.get_job_state(job_name)
            if state.exit_code != 0:
                failed_jobs.append(job_name)

        if failed_jobs:
            logger.error(f"Failed jobs: {', '.join(failed_jobs)}")
            return 1

        # Generate notebook if requested
        if self.generate_notebook:
            notebook_path = self._generate_notebook(manager)
            logger.info(f"\nNotebook generated: {notebook_path}")

        logger.info("Experiment complete!")
        return 0

    def _generate_notebook(self, manager: JobManager) -> Path:
        """Generate a Jupyter notebook with reward and SPS graphs."""
        if self.notebook_path is None:
            notebooks_dir = get_repo_root() / "experiments" / "notebooks"
            notebooks_dir.mkdir(parents=True, exist_ok=True)
            # Use .ipynb extension (will be gitignored)
            self.notebook_path = notebooks_dir / f"{self.group}.ipynb"

        # Get job states
        job_states = [manager.get_job_state(job_name) for job_name in manager.get_group_jobs(self.group)]

        # Use shared notebook generation
        generate_experiment_notebook(
            notebook_path=self.notebook_path,
            group_name=self.group,
            job_states=job_states,
        )

        return self.notebook_path
