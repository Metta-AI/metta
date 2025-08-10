"""WandB configuration for Metta AI training."""

from typing import Optional, List
from pydantic import Field

from metta.common.util.typed_config import ConfigWithBuilder


class WandbConfig(ConfigWithBuilder):
    """Configuration for WandB logging and tracking."""
    
    # Core settings
    enabled: bool = Field(default=True, description="Enable WandB logging")
    project: str = Field(default="metta", description="WandB project name")
    entity: Optional[str] = Field(default=None, description="WandB entity (team/user)")
    
    # Run identification
    group: Optional[str] = Field(default=None, description="WandB group name")
    name: Optional[str] = Field(default=None, description="WandB run name") 
    run_id: Optional[str] = Field(default=None, description="WandB run ID")
    
    # Metadata
    job_type: Optional[str] = Field(default=None, description="Job type for WandB")
    tags: List[str] = Field(default_factory=list, description="Tags for the run")
    notes: str = Field(default="", description="Notes for the run")
    
    # Data and storage
    data_dir: Optional[str] = Field(default=None, description="Data directory path")
    
    # Expected connection for validation
    expected_connection: Optional[str] = Field(default=None, description="Expected WandB connection for validation")