"""Temporary stub for MockPolicyRecord - will be fully removed after migration."""

import warnings
from ..policy_record import PolicyRecord, PolicyMetadata

warnings.warn(
    "MockPolicyRecord is deprecated and will be removed. Use SimpleCheckpointManager instead.",
    DeprecationWarning,
    stacklevel=2
)

class MockPolicyRecord(PolicyRecord):
    """Temporary stub for MockPolicyRecord - will be fully removed."""
    
    def __init__(self, uri: str = "mock://test", epoch: int = 0, score: float = 0.5):
        super().__init__(
            uri=uri,
            metadata=PolicyMetadata(epoch=epoch, score=score)
        )
    
    @classmethod
    def from_key_and_version(cls, key: str, version: int) -> "MockPolicyRecord":
        """Create a MockPolicyRecord from key and version."""
        return cls(uri=f"mock://{key}", epoch=version, score=0.5)