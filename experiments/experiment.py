"""Base class for reproducible experiments."""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import logging

from experiments.skypilot_service import get_skypilot_service
from experiments.training_job import TrainingJob, TrainingJobConfig
from metta.common.util.config import Config
from pydantic import PrivateAttr


class ExperimentConfig(Config):
    """Base configuration for all experiments."""

    name: str
    user: Optional[str] = None
    launch: bool = False
    previous_job_ids: Optional[List[str]] = None
    # output_dir: Optional[str] = Field(
    #     default=None,
    #     description="Directory to save notebook, None to skip notebook generation",
    # )

    # Store instance_name as a private attribute to avoid Pydantic validation
    _instance_name: str = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """Initialize experiment config and compute instance name."""
        super().__init__(**kwargs)
        # Compute instance name once with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._instance_name = f"{self.name}_{timestamp}"

    @property
    def instance_name(self) -> str:
        """Get the instance name for this experiment."""
        return self._instance_name


class SingleJobExperimentConfig(ExperimentConfig, TrainingJobConfig):
    """Configuration for experiments with a single training job.

    Inherits from both ExperimentConfig (for experiment metadata) and
    TrainingJobConfig (for job configuration).
    """

    pass


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.name = config.name
        self.instance_name = config.instance_name  # Use the pre-computed instance name
        self.user = config.user or os.environ.get("USER", "unknown")
        self.created_at = datetime.now().isoformat()
        self.launched_training_jobs: List[TrainingJob] = []
        self._training_job_configs: List[TrainingJobConfig] = []

    @abstractmethod
    def training_job_configs(self) -> List[TrainingJobConfig]:
        """Override this method to define the training jobs for this experiment."""
        pass

    def training_jobs(self) -> List[TrainingJob]:
        """Convert training job configs to TrainingJob objects."""
        jobs = []
        for i, config in enumerate(self.training_job_configs()):
            # Use instance_name for consistency
            job_name = (
                f"{self.instance_name}_job_{i}"
                if len(self.training_job_configs()) > 1
                else self.instance_name
            )
            jobs.append(
                TrainingJob(
                    name=job_name, config=config, instance_name=self.instance_name
                )
            )
        return jobs

    def run(self) -> Optional[str]:
        """Run the experiment and return the notebook path if generated."""
        self.load_or_launch_training_jobs()

        # TODO: Generate notebook
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
            # Show what would be launched without actually launching
            self._training_job_configs = list(self.training_job_configs())
            self.launched_training_jobs = []

            print("\n[Preview Mode - launch=False]")
            print("=" * 60)

            for i, config in enumerate(self._training_job_configs):
                print(f"\nJob {i + 1}/{len(self._training_job_configs)}:")
                print(f"  Name: {self.name}_job_{i}")
                print(f"  GPUs: {config.skypilot.gpus}")
                print(f"  Nodes: {config.skypilot.nodes}")
                print(f"  Spot: {config.skypilot.spot}")

                # Show YAML serialization details
                yaml_path, full_config = config.training.serialize_to_yaml_file(
                    instance_name=self.instance_name
                )
                print(f"\n  YAML Config Created: {yaml_path}")
                print("  Key Settings:")
                print(f"    - Curriculum: {config.training.curriculum}")
                print(f"    - Agent: {config.training.agent_config}")
                print(
                    f"    - WandB: {config.training.wandb_entity}/{config.training.wandb_project}"
                )

                if config.training.trainer:
                    print(
                        f"    - Total timesteps: {config.training.trainer.total_timesteps}"
                    )
                    print(f"    - Batch size: {config.training.trainer.batch_size}")
                    print(
                        f"    - Learning rate: {config.training.trainer.optimizer.learning_rate}"
                    )

                print(f"\n  To view full YAML: cat {yaml_path}")

                # Show local testing command
                test_command = (
                    f"uv run ./tools/train.py +experiments={self.instance_name}"
                )
                print("\n  To test locally with tools/train.py:")
                print(f"    {test_command}")

            print("\n" + "=" * 60)
            print("To launch these jobs, remove --no-launch flag")
            print("=" * 60)


class SingleJobExperiment(Experiment):
    """Base class for experiments with a single training job."""

    def __init__(self, config: SingleJobExperimentConfig):
        super().__init__(config)
        # Type narrowing for better IDE support
        self.config: SingleJobExperimentConfig = config

    def training_job_configs(self) -> List[TrainingJobConfig]:
        """Return the config itself as it already is a TrainingJobConfig."""
        # Since SingleJobExperimentConfig inherits from TrainingJobConfig,
        # we can just return it directly
        return [self.config]
