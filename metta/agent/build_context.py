"""
Build context for tracking how MettaAgents are constructed.

This module defines BuildContext which tracks construction parameters
to enable robust reconstruction of policies.
"""

import dataclasses


@dataclasses.dataclass
class BuildContext:
    """Tracks how a MettaAgent was built for reconstruction."""

    method: str  # Which builder method was used
    args: tuple = ()  # Positional arguments
    kwargs: dict = dataclasses.field(default_factory=dict)  # Keyword arguments
    source_code: dict = dataclasses.field(default_factory=dict)  # Source code for classes
    env_attributes: dict = dataclasses.field(default_factory=dict)  # Environment attributes

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "args": self.args,
            "kwargs": self.kwargs,
            "source_code": self.source_code,
            "env_attributes": self.env_attributes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BuildContext":
        """Create from dictionary."""
        return cls(**data)
