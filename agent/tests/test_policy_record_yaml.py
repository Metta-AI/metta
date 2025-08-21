#!/usr/bin/env python3
"""Test YAML serialization methods for PolicyRecord"""

import pytest
import yaml

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord


class TestPolicyRecordYAML:
    """Test YAML serialization and deserialization of PolicyRecord."""

    def test_to_yaml_basic(self):
        """Test basic to_yaml functionality."""
        metadata = PolicyMetadata(
            agent_step=100,
            epoch=5,
            generation=1,
            train_time=60.0,
        )

        policy_record = PolicyRecord(
            policy_store=None, run_name="test_policy", uri="file:///path/to/policy.pt", metadata=metadata
        )

        yaml_str = policy_record.to_yaml()

        # Verify it's valid YAML
        data = yaml.safe_load(yaml_str)
        assert isinstance(data, dict)
        assert data["run_name"] == "test_policy"
        assert data["uri"] == "file:///path/to/policy.pt"
        assert data["metadata"]["agent_step"] == 100
        assert data["metadata"]["epoch"] == 5
        assert data["metadata"]["generation"] == 1
        assert data["metadata"]["train_time"] == 60.0

    def test_from_yaml_basic(self):
        """Test basic from_yaml functionality."""
        yaml_str = """
run_name: test_policy
uri: file:///path/to/policy.pt
metadata:
  agent_step: 100
  epoch: 5
  generation: 1
  train_time: 60.0
  score: 0.85
"""

        policy_record = PolicyRecord.from_yaml(yaml_str)

        assert policy_record.run_name == "test_policy"
        assert policy_record.uri == "file:///path/to/policy.pt"
        assert policy_record.metadata["agent_step"] == 100
        assert policy_record.metadata["epoch"] == 5
        assert policy_record.metadata["generation"] == 1
        assert policy_record.metadata["train_time"] == 60.0
        assert policy_record.metadata["score"] == 0.85

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization preserves data."""
        metadata = PolicyMetadata(
            agent_step=100, epoch=5, generation=1, train_time=60.0, score=0.85, additional_field="test_value"
        )

        original_record = PolicyRecord(
            policy_store=None, run_name="test_policy_v1", uri="file:///path/to/policy.pt", metadata=metadata
        )

        # Serialize to YAML
        yaml_str = original_record.to_yaml()

        # Deserialize from YAML
        reconstructed_record = PolicyRecord.from_yaml(yaml_str)

        # Verify all fields match (excluding policy_store which is None)
        assert original_record.run_name == reconstructed_record.run_name
        assert original_record.uri == reconstructed_record.uri
        assert dict(original_record.metadata) == dict(reconstructed_record.metadata)

    def test_from_yaml_missing_required_fields(self):
        """Test that from_yaml raises ValueError for missing required fields."""
        # Missing run_name
        yaml_str = """
uri: file:///path/to/policy.pt
metadata:
  agent_step: 100
  epoch: 5
  generation: 1
  train_time: 60.0
"""
        with pytest.raises(ValueError, match="Missing required field: run_name"):
            PolicyRecord.from_yaml(yaml_str)

        # Missing metadata
        yaml_str = """
run_name: test_policy
uri: file:///path/to/policy.pt
"""
        with pytest.raises(ValueError, match="Missing required field: metadata"):
            PolicyRecord.from_yaml(yaml_str)

    def test_from_yaml_invalid_yaml(self):
        """Test that from_yaml handles invalid YAML gracefully."""
        with pytest.raises(yaml.YAMLError):
            PolicyRecord.from_yaml("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="YAML must represent a dictionary"):
            PolicyRecord.from_yaml("not_a_dict")

    def test_to_yaml_excludes_policy_store_and_policy(self):
        """Test that to_yaml excludes policy_store and policy components."""
        metadata = PolicyMetadata(
            agent_step=100,
            epoch=5,
            generation=1,
            train_time=60.0,
        )

        policy_record = PolicyRecord(
            policy_store=None, run_name="test_policy", uri="file:///path/to/policy.pt", metadata=metadata
        )

        yaml_str = policy_record.to_yaml()
        data = yaml.safe_load(yaml_str)

        # Verify only expected fields are present
        expected_keys = {"run_name", "uri", "metadata"}
        assert set(data.keys()) == expected_keys

        # Verify no policy_store or policy-related fields
        assert "policy_store" not in data
        assert "policy" not in data
        assert "_cached_policy" not in data
        assert "_policy_store" not in data

    def test_from_yaml_with_policy_store(self):
        """Test that from_yaml can accept a policy_store parameter."""
        yaml_str = """
run_name: test_policy
uri: file:///path/to/policy.pt
metadata:
  agent_step: 100
  epoch: 5
  generation: 1
  train_time: 60.0
"""

        # Mock policy store (we can't easily create a real one in tests)
        mock_policy_store = None

        policy_record = PolicyRecord.from_yaml(yaml_str, policy_store=mock_policy_store)

        assert policy_record.run_name == "test_policy"
        assert policy_record._policy_store is mock_policy_store
