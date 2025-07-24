"""Common types for experiments."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class TrainingJob:
    """Represents a launched training job with its identifiers."""
    wandb_run_id: str
    skypilot_job_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wandb_run_id": self.wandb_run_id,
            "skypilot_job_id": self.skypilot_job_id,
            "config": self.config,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            wandb_run_id=data["wandb_run_id"],
            skypilot_job_id=data.get("skypilot_job_id"),
            config=data.get("config"),
            notes=data.get("notes"),
            timestamp=timestamp
        )