"""Temporary stub for PolicyStore - will be fully removed after migration."""

import warnings
from typing import Any, Dict, Optional, List
from pathlib import Path

from .policy_record import PolicyRecord

warnings.warn(
    "PolicyStore is deprecated and will be removed. Use SimpleCheckpointManager instead.",
    DeprecationWarning, 
    stacklevel=2
)

class PolicyStore:
    """Temporary stub for PolicyStore - will be fully removed."""
    
    def __init__(self, *args, **kwargs):
        pass  # Allow creation for compatibility but don't do anything
    
    @classmethod
    def create(cls, *args, **kwargs) -> "PolicyStore":
        """Create a PolicyStore stub for compatibility."""
        warnings.warn(
            "PolicyStore.create() is deprecated. Use SimpleCheckpointManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return cls()
    
    @staticmethod
    def from_file(*args, **kwargs) -> "PolicyStore":
        raise NotImplementedError("PolicyStore is deprecated. Use SimpleCheckpointManager instead.")
        
    def search(self, *args, **kwargs) -> List[PolicyRecord]:
        return []
        
    def add_policy_record(self, *args, **kwargs) -> PolicyRecord:
        raise NotImplementedError("PolicyStore is deprecated. Use SimpleCheckpointManager instead.")
        
    def policy_record_or_mock(self, policy_uri: str, selector_type: str = "latest") -> PolicyRecord:
        """Return a mock PolicyRecord for compatibility."""
        from .mocks.mock_policy_record import MockPolicyRecord
        return MockPolicyRecord(uri=policy_uri or "mock://default")