"""Base class for reproducible experiments."""

import os
from abc import ABC
from datetime import datetime
from typing import List, Optional

import logging

from experiments.skypilot_service import get_skypilot_service
from experiments.training_job import TrainingJob, TrainingJobConfig
from experiments.notebooks.notebook import write_notebook, NotebookConfig
from metta.common.util.config import Config
from pydantic import Field


class ExperimentConfig(Config):
    """Base configuration for all experiments."""

    name: str
    user: Optional[str] = None
    launch: bool = False
    previous_job_ids: Optional[List[str]] = None
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save notebook, None to skip notebook generation",
    )
    notebook: Optional[NotebookConfig] = Field(
        default_factory=lambda: NotebookConfig(simplified=True)
    )


class SingleJobExperimentConfig(ExperimentConfig, TrainingJobConfig):
    """Configuration for experiments with a single training job."""

    pass


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.name = config.name
        self.user = config.user or os.environ.get("USER", "unknown")
        self.created_at = datetime.now().isoformat()
        self.launched_training_jobs: List[TrainingJob] = []
        self._training_job_configs: List[TrainingJobConfig] = []

    @property
    def training_job_configs(self) -> List[TrainingJobConfig]:
        """Override this property to define the training jobs for this experiment."""
        return []

    def training_jobs(self) -> List[TrainingJob]:
        """Convert training job configs to TrainingJob objects."""
        jobs = []
        for i, config in enumerate(self.training_job_configs):
            job_name = f"{self.name}_job_{i}"
            jobs.append(TrainingJob(name=job_name, config=config))
        return jobs

    def run(self) -> Optional[str]:
        """Run the experiment and return the notebook path if generated."""
        self.load_or_launch_training_jobs()

        # Only generate notebook if output_dir is specified
        if self.config.output_dir is not None:
            return self.generate_notebook()
        else:
            log = logging.getLogger(__name__)
            log.info("Skipping notebook generation (output_dir is None)")
            return None

    def load_training_jobs(self) -> List[TrainingJob]:
        """Load training jobs from previous job IDs."""

        training_jobs = self.training_jobs()
        if len(training_jobs) != len(self.config.previous_job_ids):
            raise ValueError(
                f"Number of training jobs ({len(training_jobs)}) does not match number of previous job IDs ({len(self.config.previous_job_ids)})"
            )
        for i, job in enumerate(training_jobs):
            job.launched = True
            job.success = True
            job.job_id = self.config.previous_job_ids[i]
            service = get_skypilot_service()
            job.name = service.get_wandb_run_name_from_sky_job(job.job_id) or job.name

        self.launched_training_jobs = training_jobs
        self._training_job_configs = []
        return training_jobs

    def launch_training_jobs(self) -> List[TrainingJob]:
        """Launch all training jobs in the experiment."""
        log = logging.getLogger(__name__)
        jobs = self.training_jobs()

        if not jobs:
            print("No jobs to launch")
            return []

        print(f"\nLaunching {len(jobs)} training job(s)...")

        for i, job in enumerate(jobs):
            print(f"\n[{i + 1}/{len(jobs)}] Launching {job.name}...")
            start_time = datetime.now()

            success = job.launch()

            elapsed = (datetime.now() - start_time).total_seconds()
            if success:
                self.launched_training_jobs.append(job)
                print(f"✓ Successfully launched {job.name} (took {elapsed:.1f}s)")
                if job.job_id:
                    print(f"  Job ID: {job.job_id}")
            else:
                log.error(f"Failed to launch training job: {job.name}")
                print(f"✗ Failed to launch {job.name} (took {elapsed:.1f}s)")

        self.launched_training_jobs = jobs
        self._training_job_configs = []

        print(
            f"\nLaunch complete: {len([j for j in jobs if j.launched])} succeeded, {len([j for j in jobs if not j.launched])} failed"
        )

        return jobs

    def load_or_launch_training_jobs(self):
        """Load or launch training jobs."""
        if self.config.launch:
            self.launch_training_jobs()
        elif self.config.previous_job_ids:
            self.load_training_jobs()
        else:
            # Store the configs for potential later launching
            self._training_job_configs = list(self.training_job_configs)
            self.launched_training_jobs = []

    def generate_notebook(self) -> str:
        """Generate a notebook and return its path."""
        log = logging.getLogger(__name__)

        # Convert NotebookConfig to sections list
        sections = []
        if self.config.notebook:
            nb_config = self.config.notebook
            if nb_config.simplified:
                # Simplified notebook with just config & monitor
                sections = ["setup", "state", "config", "monitor"]
            else:
                # Full notebook with all requested sections
                if nb_config.setup:
                    sections.append("setup")
                if nb_config.state:
                    sections.append("state")
                if nb_config.launch:
                    sections.append("launch")
                if nb_config.monitor:
                    sections.append("monitor")
                if nb_config.analysis:
                    sections.append("analysis")
                if nb_config.replays:
                    sections.append("replays")
                if nb_config.scratch:
                    sections.append("scratch")
                if nb_config.export:
                    sections.append("export")

        notebook_path = write_notebook(
            user=self.user,
            name=self.name,
            launched_jobs=self.launched_training_jobs,
            training_job_configs=self._training_job_configs,
            output_dir=self.config.output_dir,
            sections=sections if sections else None,
        )

        print(f"Notebook saved to: {notebook_path}")
        return notebook_path


class SingleJobExperiment(Experiment):
    """Base class for experiments with a single training job."""

    def __init__(self, config: SingleJobExperimentConfig):
        super().__init__(config)
        # Type narrowing for better IDE support
        self.config: SingleJobExperimentConfig = config

    @property
    def training_job_configs(self) -> List[TrainingJobConfig]:
        """Create a single training job config from the experiment config."""
        # Extract only the TrainingJobConfig fields from the combined config
        training_config = TrainingJobConfig(
            curriculum=self.config.curriculum,
            gpus=self.config.gpus,
            nodes=self.config.nodes,
            spot=self.config.spot,
            git_check=self.config.git_check,
            wandb_tags=self.config.wandb_tags,
            additional_args=self.config.additional_args,
        )
        return [training_config]
