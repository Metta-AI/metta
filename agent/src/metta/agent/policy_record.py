"""
PolicyRecord: A lightweight wrapper around PolicyAgent that delegates most functionality to the agent.
"""

import logging

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_metadata import PolicyMetadata

logger = logging.getLogger(__name__)


class PolicyRecord:
    """A lightweight wrapper around PolicyAgent that delegates most functionality to the agent."""

    def __init__(
        self,
        run_name: str,
        uri: str | None,
        metadata: PolicyMetadata | dict,
        policy: "PolicyAgent",
        wandb_entity: str | None = None,  # for loading policies from wandb
        wandb_project: str | None = None,  # for loading policies from wandb
    ):
        # Set policy directly - must be a PolicyAgent
        if not isinstance(policy, PolicyAgent):
            raise TypeError(f"policy must be a PolicyAgent, got {type(policy).__name__}")
        self._cached_policy: "PolicyAgent" = policy

        # Set PolicyRecord properties on the policy
        self._cached_policy.set_policy_record_properties(
            run_name=run_name,
            uri=uri,
            metadata=metadata,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )

    # Delegation properties - delegate to policy for backwards compatibility
    @property
    def run_name(self) -> str:
        """Get run name - delegates to policy."""
        return self._cached_policy.run_name

    @property
    def uri(self) -> str | None:
        """Get URI - delegates to policy."""
        return self._cached_policy.uri

    @property
    def wandb_entity(self) -> str | None:
        """Get wandb entity - delegates to policy."""
        return self._cached_policy.wandb_entity

    @property
    def wandb_project(self) -> str | None:
        """Get wandb project - delegates to policy."""
        return self._cached_policy.wandb_project

    def extract_wandb_run_info(self) -> tuple[str, str, str, str | None]:
        """Extract wandb run info - delegates to policy."""
        return self._cached_policy.extract_wandb_run_info()

    @property
    def metadata(self) -> PolicyMetadata:
        """Get metadata - delegates to policy with backwards compatibility."""
        if hasattr(self._cached_policy, "metadata"):
            return self._cached_policy.metadata
        elif hasattr(self._cached_policy, "_metadata"):
            return self._cached_policy._metadata
        else:
            # Handle backwards compatibility for old PolicyRecord instances
            if not hasattr(self, "_metadata"):
                # Try backwards compatibility names
                old_metadata_names = ["checkpoint"]
                for name in old_metadata_names:
                    if hasattr(self, name):
                        logger.warning(
                            f"Found metadata under old attribute name '{name}'. "
                            f"This PolicyRecord was saved with an older version of the code. "
                            f"Converting to new format."
                        )
                        # Convert old metadata to new format
                        old_metadata = getattr(self, name)
                        if isinstance(old_metadata, PolicyMetadata):
                            self._cached_policy._metadata = old_metadata
                        elif isinstance(old_metadata, dict):
                            self._cached_policy._metadata = PolicyMetadata(**old_metadata)
                        else:
                            raise TypeError(
                                f"Old metadata must be PolicyMetadata or dict, got {type(old_metadata).__name__}"
                            )
                        return self._cached_policy._metadata

                # If no old names found, collect available attributes
                available_attrs = {}
                for attr in dir(self):
                    if attr == "metadata":  # Skip this property to avoid recursion
                        continue
                    if not attr.startswith("_"):
                        try:
                            value = getattr(self, attr)
                            if not callable(value):
                                available_attrs[attr] = type(value).__name__
                        except Exception as e:
                            available_attrs[attr] = f"<Error accessing: {e}>"

                raise AttributeError(
                    f"No metadata found under any known attribute names. "
                    f"Available attributes: {available_attrs}. "
                    f"This PolicyRecord may be corrupted or from an incompatible version."
                )
            # This should not happen in the new architecture, but handle gracefully
            fallback_metadata = getattr(self, "_metadata", None)
            if isinstance(fallback_metadata, PolicyMetadata):
                return fallback_metadata
            else:
                return PolicyMetadata()

    @property
    def file_path(self) -> str:
        """Extract file path from URI - delegates to policy."""
        return self._cached_policy.file_path

    @property
    def policy(self) -> "PolicyAgent":
        """Get the policy."""
        return self._cached_policy

    @property
    def cached_policy(self) -> "PolicyAgent":
        """Get the cached policy."""
        return self._cached_policy

    @cached_policy.setter
    def cached_policy(self, policy: "PolicyAgent") -> None:
        """Set the cached policy directly."""
        if not isinstance(policy, PolicyAgent):
            raise TypeError(f"cached_policy must be a PolicyAgent, got {type(policy).__name__}")
        self._cached_policy = policy

    def num_params(self) -> int:
        """Count trainable parameters - delegates to policy."""
        return self._cached_policy.num_params()

    def __repr__(self):
        """Generate detailed representation - delegates to policy."""
        # Get the policy's representation and modify it to show as PolicyRecord
        policy_repr = str(self._cached_policy)
        # Replace "MettaAgent(" with "PolicyRecord(" to maintain the expected interface
        return policy_repr.replace("MettaAgent(", "PolicyRecord(", 1)
