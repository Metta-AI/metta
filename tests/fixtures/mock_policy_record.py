# tests/conftest.py
from typing import Tuple


class MockPolicyRecord:
    """Mock implementation of PolicyRecord for testing."""

    def __init__(self, policy_key: str, policy_version: int):
        self._policy_key = policy_key
        self._policy_version = policy_version

    def wandb_key_and_version(self) -> Tuple[str, int]:
        """Return the policy key and version as a tuple."""
        return self._policy_key, self._policy_version
