"""Base class for reproducible experiments."""

import os
from abc import ABC
from datetime import datetime
from typing import List, Optional

import logging

from experiments.skypilot_service import get_skypilot_service
from experiments.training_job import TrainingJob, TrainingJobConfig
from experiments.skypilot_job_config import SkypilotJobConfig
from experiments.training_run_config import TrainingRunConfig
from metta.common.util.config import Config


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


class SingleJobExperimentConfig(ExperimentConfig):
    """Configuration for experiments with a single training job.

    Composes SkypilotJobConfig and TrainingRunConfig without duplicating defaults.
    """

    # Compose the two configs without redefining defaults
    skypilot: SkypilotJobConfig = SkypilotJobConfig()
    training: TrainingRunConfig = TrainingRunConfig()

    # Optional overrides for common parameters (only if user wants to change them)
    total_timesteps: Optional[int] = None
    num_workers: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

    def to_training_job_config(self) -> TrainingJobConfig:
        """Convert to a properly structured TrainingJobConfig."""
        # Start with the composed configs
        config = TrainingJobConfig(skypilot=self.skypilot, training=self.training)

        # Apply any user overrides for trainer parameters
        if any(
            [
                self.total_timesteps,
                self.num_workers,
                self.batch_size,
                self.learning_rate,
            ]
        ):
            from metta.rl.trainer_config import (
                TrainerConfig,
                OptimizerConfig,
                CheckpointConfig,
                SimulationConfig,
                TorchProfilerConfig,
            )

            # Get existing trainer or create new one
            if config.training.trainer:
                trainer_dict = config.training.trainer.model_dump()
            else:
                # Create minimal trainer config with required fields
                trainer_dict = {
                    "checkpoint": CheckpointConfig(
                        checkpoint_dir="${run_dir}/checkpoints"
                    ).model_dump(),
                    "simulation": SimulationConfig(
                        replay_dir="${run_dir}/replays"
                    ).model_dump(),
                    "profiler": TorchProfilerConfig(
                        profile_dir="${run_dir}/torch_traces"
                    ).model_dump(),
                    "curriculum": config.training.curriculum,
                    "num_workers": 1,  # Required field, use default
                }

            # Apply overrides
            if self.total_timesteps is not None:
                trainer_dict["total_timesteps"] = self.total_timesteps
            if self.num_workers is not None:
                trainer_dict["num_workers"] = self.num_workers
            if self.batch_size is not None:
                trainer_dict["batch_size"] = self.batch_size
                # Auto-calculate minibatch_size if not specified
                if "minibatch_size" not in trainer_dict:
                    trainer_dict["minibatch_size"] = min(512, self.batch_size // 4)
            if self.learning_rate is not None:
                trainer_dict["optimizer"] = OptimizerConfig(
                    learning_rate=self.learning_rate
                ).model_dump()

            config.training.trainer = TrainerConfig(**trainer_dict)

        return config


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
            self._training_job_configs = list(self.training_job_configs)
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
                yaml_path, full_config = config.training.serialize_to_yaml_file()
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
                test_command = config.training.save_for_local_testing()
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

    @property
    def training_job_configs(self) -> List[TrainingJobConfig]:
        """Create a single training job config from the experiment config."""
        # Convert the experiment config to a TrainingJobConfig
        return [self.config.to_training_job_config()]
