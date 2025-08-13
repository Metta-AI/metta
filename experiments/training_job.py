"""Training job configuration and management."""

from typing import Optional
from datetime import datetime

from metta.common.util.config import Config
from experiments.skypilot_job_config import SkypilotJobConfig
from experiments.training_run_config import TrainingRunConfig


class TrainingJobConfig(Config):
    """Complete configuration for a training job.

    This combines both the Skypilot infrastructure configuration
    and the training run configuration.
    """

    skypilot: SkypilotJobConfig = SkypilotJobConfig()
    training: TrainingRunConfig = TrainingRunConfig()


class TrainingJob:
    """Represents a launched training job with its identifiers."""

    def __init__(self, name: str, config: Optional[TrainingJobConfig] = None):
        """Initialize a training job.

        Args:
            name: Name for the training job
            config: Complete job configuration
        """
        self.name = name
        self.config = config or TrainingJobConfig()
        self.job_id: Optional[str] = None
        self.launched: bool = False
        self.success: bool = False
        self.cancelled: bool = False
        self.launch_time: Optional[datetime] = None
        self.timestamp = datetime.now()

        # Additional metadata
        self.notes: str = ""

    def launch(self):
        """Launch the training job via Skypilot."""
        if self.launched:
            raise ValueError(f"Training job {self.name} has already been launched")

        # Use the skypilot service
        from experiments.skypilot_service import get_skypilot_service

        service = get_skypilot_service()
        service.launch_training(
            run_name=self.name,
            training_job=self,  # Pass self with full config
        )

        # Job state is updated by the service
        return self.success
