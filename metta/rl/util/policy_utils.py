"""Policy-related utility functions."""

from typing import Any, Optional

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord


def create_initial_policy_record(
    policy_store: Any,
    initial_policy_uri: Optional[str],
    initial_generation: int,
) -> Optional[PolicyRecord]:
    """Create a minimal PolicyRecord for stats tracking.

    This is used when we need to track the initial policy information
    for stats and metadata purposes.

    Args:
        policy_store: The policy store instance
        initial_policy_uri: URI of the initial policy (can be None)
        initial_generation: Generation number of the initial policy

    Returns:
        PolicyRecord if initial_policy_uri exists, None otherwise
    """
    if not initial_policy_uri:
        return None

    metadata = PolicyMetadata(generation=initial_generation)
    return PolicyRecord(policy_store=policy_store, run_name="", uri=initial_policy_uri, metadata=metadata)
