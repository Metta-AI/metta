from typing import List, Optional


from metta.common.util.config import Config
from datetime import datetime


class TrainingJobConfig(Config):
    """Configuration for a training job."""

    # Core launch parameters (matching launch.py argument names)
    curriculum: str = "invalid"
    gpus: int = 1
    nodes: int = 1
    spot: bool = True
    git_check: bool = True  # Changed from skip_git_check, inverted default
    wandb_tags: Optional[List[str]] = None

    # Additional arguments passed to trainer (e.g., trainer.optimizer.learning_rate=0.001)
    additional_args: Optional[List[str]] = None

    def get_arg_value(self, key: str) -> Optional[str]:
        """Get value of a specific additional arg by key."""
        if not self.additional_args:
            return None
        for arg in self.additional_args:
            if "=" in arg:
                arg_key, arg_value = arg.split("=", 1)
                if arg_key == key:
                    return arg_value
        return None


class TrainingJob:
    """Represents a launched training job with its identifiers."""

    name: str
    job_id: Optional[str] = None
    config: TrainingJobConfig
    launched: bool = False
    success: bool = False
    cancelled: bool = False
    launch_time: Optional["datetime"] = None

    # Additional metadata that can be attached
    notes: str = ""
    timestamp: Optional["datetime"] = None

    def __init__(self, name: str, config: Optional[TrainingJobConfig] = None):
        self.name = name
        self.config = config or TrainingJobConfig()
        from datetime import datetime

        self.timestamp = datetime.now()

    def launch(self):
        if self.launched:
            raise ValueError(f"Training job {self.name} has already been launched")

        # Use the skypilot service
        from experiments.skypilot_service import get_skypilot_service

        service = get_skypilot_service()
        result = service.launch_training(
            run_name=self.name,
            curriculum=self.config.curriculum,
            gpus=self.config.gpus,
            nodes=self.config.nodes,
            spot=self.config.spot,
            skip_git_check=not self.config.git_check,  # Invert git_check to skip_git_check
            wandb_tags=self.config.wandb_tags,
            additional_args=self.config.additional_args,
            training_job=self,  # Pass self to be tracked
        )

        # Job state is updated by the service
        return self.success
