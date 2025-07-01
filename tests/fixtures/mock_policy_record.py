# tests/conftest.py

from metta.agent.policy_metadata import PolicyMetadata


class MockPolicyRecord:
    """Mock implementation of PolicyRecord for testing."""

    def __init__(self, policy_key: str, policy_version: int):
        self.uri = policy_key
        self.metadata = PolicyMetadata()
        self.metadata.epoch = policy_version
