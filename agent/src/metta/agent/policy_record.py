"""Temporary stub for PolicyRecord - will be fully removed after migration."""

import warnings
from typing import Any, Dict, Optional
from dataclasses import dataclass

warnings.warn(
    "PolicyRecord is deprecated and will be removed. Use SimpleCheckpointManager instead.",
    DeprecationWarning,
    stacklevel=2
)

@dataclass
class PolicyMetadata:
    """Temporary stub for PolicyMetadata."""
    epoch: Optional[int] = None
    score: Optional[float] = None
    
    def __post_init__(self):
        pass

@dataclass 
class PolicyRecord:
    """Temporary stub for PolicyRecord - will be fully removed."""
    uri: Optional[str] = None
    metadata: PolicyMetadata = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = PolicyMetadata()