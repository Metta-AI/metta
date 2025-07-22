"""Configuration for multi-agent training."""

from typing import Optional, Literal
from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class MultiAgentConfig(BaseModelWithForbidExtra):
    """Configuration for multi-agent training."""
    
    # Core settings
    num_policies: int = Field(gt=0, description="Number of distinct policies to train")
    
    # Reward strategy
    reward_mode: Literal["competition", "collaboration"] = "competition"
    reward_sharing_fn: Optional[str] = None  # For collaboration mode
    
    # Evolution strategy  
    evolution_mode: Literal["stable", "evolving"] = "stable"
    mutation_rate: float = Field(default=0.01, ge=0, le=1)
    selection_pressure: float = Field(default=2.0, gt=0)
    crossover_rate: float = Field(default=0.1, ge=0, le=1)
    evolution_interval: int = Field(default=100, gt=0)
    
    # Diversity
    diversity_metric: Optional[Literal["weight_l2", "weight_cosine", "behavioral"]] = None
    diversity_weight: float = Field(default=0.0, ge=0)
    behavioral_state_encoder: Optional[str] = None  # Path to encoder
    
    # Policy assignment
    policy_mapping: Optional[list[int]] = None  # For stable mode
    
    # Training coordination
    shared_experience: bool = False  # Whether policies share experience
    synchronized_updates: bool = True  # Whether to sync policy updates