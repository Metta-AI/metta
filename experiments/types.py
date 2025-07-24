"""Common types for experiments."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field, ConfigDict

from metta.common.util.config import Config


class TrainingJobConfig(Config):
    """Configuration for a training job."""
    model_config = ConfigDict(extra="forbid")
    
    # Core launch parameters (matching launch.py argument names)
    curriculum: str = Field(..., description="Path to curriculum config")
    gpus: int = Field(1, description="Number of GPUs per node")
    nodes: int = Field(1, description="Number of nodes")
    no_spot: bool = Field(False, description="Whether to disable spot instances")
    wandb_tags: Optional[List[str]] = Field(None, description="Tags for wandb")
    
    # Additional arguments passed to trainer (e.g., trainer.optimizer.learning_rate=0.001)
    additional_args: List[str] = Field(default_factory=list, description="Additional command line arguments")
    
    def get_arg_value(self, key: str) -> Optional[str]:
        """Get value of a specific additional arg by key."""
        for arg in self.additional_args:
            if "=" in arg:
                arg_key, arg_value = arg.split("=", 1)
                if arg_key == key:
                    return arg_value
        return None


@dataclass
class TrainingJob:
    """Represents a launched training job with its identifiers."""
    wandb_run_name: str  # The WandB run name (e.g., "user.experiments.arena.12-04")
    skypilot_job_id: Optional[str] = None
    config: Optional[TrainingJobConfig] = None
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wandb_run_name": self.wandb_run_name,
            "skypilot_job_id": self.skypilot_job_id,
            "config": self.config.model_dump() if self.config else None,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        config = None
        if data.get("config"):
            if isinstance(data["config"], dict):
                config = TrainingJobConfig(**data["config"])
            else:
                config = data["config"]  # Already a TrainingJobConfig
        
        return cls(
            wandb_run_name=data.get("wandb_run_name") or data.get("wandb_run_id"),  # Support old format
            skypilot_job_id=data.get("skypilot_job_id"),
            config=config,
            notes=data.get("notes"),
            timestamp=timestamp
        )