from unittest.mock import Mock

import pytest
import torch

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_state import PolicyState


@pytest.mark.hourly
def test_policy_metadata_initialization():
    """Critical test for policy metadata initialization - runs hourly."""
    # Test with required fields
    metadata = PolicyMetadata(
        agent_step=1000,
        epoch=10,
        generation=5,
        train_time=3600.0,
    )

    # Test attribute access
    assert metadata.agent_step == 1000
    assert metadata.epoch == 10
    assert metadata.generation == 5
    assert metadata.train_time == 3600.0

    # Test with additional fields
    metadata_with_extras = PolicyMetadata(
        agent_step=2000,
        epoch=20,
        generation=10,
        train_time=7200.0,
        wandb_run_id="test-run-123",
        score=0.95,
    )

    assert metadata_with_extras.wandb_run_id == "test-run-123"
    assert metadata_with_extras.score == 0.95


@pytest.mark.hourly
def test_policy_record_initialization():
    """Critical test for policy record initialization - runs hourly."""
    # Create mock policy store
    mock_policy_store = Mock()

    metadata = PolicyMetadata(
        agent_step=2000,
        epoch=20,
        generation=10,
        train_time=7200.0,
    )

    record = PolicyRecord(
        policy_store=mock_policy_store,
        run_name="test-run",
        uri="file:///path/to/checkpoint.pt",
        metadata=metadata,
    )

    assert record.run_name == "test-run"
    assert record.uri == "file:///path/to/checkpoint.pt"
    assert record.metadata == metadata
    assert record.file_path == "/path/to/checkpoint.pt"


@pytest.mark.hourly
def test_policy_state_operations():
    """Critical test for policy state tensor operations - runs hourly."""
    # Test empty state
    state = PolicyState()
    assert state.lstm_h is None
    assert state.lstm_c is None
    assert state.hidden is None

    # Test with tensors
    h = torch.randn(2, 3, 128)
    c = torch.randn(2, 3, 128)
    state = PolicyState(lstm_h=h, lstm_c=c)

    assert state.lstm_h is not None
    assert state.lstm_c is not None
    assert state.lstm_h.shape == (2, 3, 128)
    assert state.lstm_c.shape == (2, 3, 128)

    # Test device transfer
    if torch.cuda.is_available():
        cuda_state = state.to("cuda")
        assert cuda_state.lstm_h.device.type == "cuda"
        assert cuda_state.lstm_c.device.type == "cuda"


@pytest.mark.daily
@pytest.mark.integration
def test_policy_loading_from_wandb():
    """Integration test for loading policies from WandB - runs daily."""
    # This would test actual WandB integration
    # Placeholder for actual implementation
    pass


@pytest.mark.daily
def test_policy_checkpoint_persistence():
    """Comprehensive test for policy checkpoint save/load - runs daily."""
    # This would test full checkpoint persistence
    # Placeholder for actual implementation
    pass
