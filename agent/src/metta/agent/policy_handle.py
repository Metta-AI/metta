"""
This file implements a PolicyHandle class that provides a lazy loading mechanism for policy records.
"""

from typing import Callable

from metta.agent.policy_loader import PolicyLoader
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord


class PolicyHandle:
    """A handle to a policy record that supports lazy loading.

    This class provides a way to reference a policy without immediately loading it,
    allowing for deferred loading when the policy is actually needed.
    """

    def __init__(
        self,
        uri: str,
        factory: Callable[[PolicyLoader], PolicyRecord],
        run_name: str | None = None,
        metadata: PolicyMetadata | dict | None = None,
    ):
        """Initialize a PolicyHandle.

        Args:
            uri: The URI identifier for the policy
            factory: A callable that takes a PolicyLoader and returns the PolicyRecord
            run_name: Optional name for the policy run
            metadata: Optional metadata for the policy (PolicyMetadata or dict)
        """
        self.uri: str = uri
        self.factory: Callable[[PolicyLoader], PolicyRecord] = factory
        self.run_name: str = run_name if run_name is not None else uri
        # Use the setter to ensure proper type
        self.metadata = metadata

    @property
    def metadata(self) -> PolicyMetadata | None:
        """Get the metadata."""
        return getattr(self, "_metadata", None)

    @metadata.setter
    def metadata(self, value: PolicyMetadata | dict | None) -> None:
        """Set metadata, ensuring it's a PolicyMetadata instance."""
        if value is None:
            self._metadata = None
        elif isinstance(value, PolicyMetadata):
            self._metadata = value
        elif isinstance(value, dict):
            # Automatically convert dict to PolicyMetadata
            self._metadata = PolicyMetadata(**value)
        else:
            raise TypeError(f"metadata must be PolicyMetadata, dict, or None, got {type(value).__name__}")

    def load(self, policy_loader: PolicyLoader) -> PolicyRecord:
        """Load the policy record using the provided PolicyLoader."""

        return self.factory(policy_loader)
