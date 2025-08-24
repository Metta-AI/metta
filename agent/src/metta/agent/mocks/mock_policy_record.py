import torch.nn as nn
from typing_extensions import override

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore

from .mock_agent import MockAgent
from .mock_policy import MockPolicy

MockPolicyMetadata = PolicyMetadata()
MockPolicyStore = PolicyStore()


class MockPolicyRecord(PolicyRecord):
    """Mock implementation of PolicyRecord for testing."""

    def __init__(
        self,
        run_name: str = "mock_run",
        uri: str | None = "mock://policy",
        metadata: PolicyMetadata = MockPolicyMetadata,
        policy: "PolicyAgent | None" = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ):
        """Initialize a mock policy record.

        Args:
            run_name: Run name (defaults to "mock_run")
            uri: Policy URI (defaults to "mock://policy")
            metadata: PolicyMetadata instance (creates default if None)
            policy: Optional policy agent
            wandb_entity: Optional wandb entity
            wandb_project: Optional wandb project
        """

        super().__init__(run_name, uri, metadata, policy, wandb_entity, wandb_project)

    @classmethod
    def from_key_and_version(
        cls,
        policy_key: str,
        policy_version: int,
        policy_store: PolicyStore = MockPolicyStore,
        run_name: str = "test_run",
    ) -> "MockPolicyRecord":
        """Create a MockPolicyRecord from policy key and version.

        Args:
            policy_key: The policy key/URI
            policy_version: The policy version/epoch
            policy_store: Optional policy store (defaults to None for tests)
            run_name: Optional run name (defaults to "test_run")

        Returns:
            A new MockPolicyRecord instance
        """
        # Create metadata with the version
        metadata = PolicyMetadata()
        metadata.epoch = policy_version

        # Create instance using parent's __init__
        instance = cls(policy_store=policy_store, run_name=run_name, uri=policy_key, metadata=metadata)

        return instance

    @property
    @override
    def policy(self) -> nn.Module:
        """Override policy property for mock behavior."""
        # Your custom implementation here
        # For example, return a mock policy directly:
        if self.uri is None:
            # return a fake agent that always outputs no action
            if not hasattr(self, "_mock_agent"):
                self._mock_agent = MockAgent()
            return self._mock_agent
        else:
            return MockPolicy()
