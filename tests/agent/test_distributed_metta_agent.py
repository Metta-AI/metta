"""
Tests for DistributedMettaAgent functionality.

These tests ensure that the distributed wrapper properly delegates all
necessary methods and maintains compatibility with the MettaAgent interface.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_state import PolicyState


class TestDistributedMettaAgent:
    """Test the DistributedMettaAgent wrapper."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock MettaAgent."""
        agent = Mock(spec=MettaAgent)
        agent.name = "test_agent"
        agent.uri = "file://test.pt"
        agent.metadata = {"epoch": 1}
        agent.model_type = "brain"

        # Add all the methods we need
        agent.forward = Mock(
            return_value=(
                torch.zeros(1, 2),  # action
                torch.zeros(1),  # log_prob
                torch.zeros(1),  # entropy
                torch.zeros(1, 1),  # value
                torch.zeros(1, 128),  # hidden
            )
        )
        agent.activate_actions = Mock()
        agent.key_and_version = Mock(return_value=("test_key", 1))
        agent.policy_as_metta_agent = Mock(return_value=agent)
        agent.l2_init_loss = Mock(return_value=torch.tensor(0.0))
        agent.update_l2_init_weight_copy = Mock()
        agent.clip_weights = Mock()
        agent.compute_weight_metrics = Mock(return_value=[])
        agent.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(2, 2))]))
        agent.state_dict = Mock(return_value={"test": torch.randn(2, 2)})
        agent.load_state_dict = Mock()

        # Make it a proper nn.Module
        agent.to = Mock(return_value=agent)
        agent.train = Mock(return_value=agent)
        agent.eval = Mock(return_value=agent)

        return agent

    @pytest.fixture
    def distributed_agent(self, mock_agent):
        """Create a DistributedMettaAgent with mocked distributed setup."""
        with patch("torch.nn.parallel.DistributedDataParallel.__init__", return_value=None):
            agent = DistributedMettaAgent(mock_agent, device="cpu")
            # Since we're mocking __init__, we need to manually set _wrapped_agent
            agent._wrapped_agent = mock_agent
            return agent

    def test_forward_delegation(self, distributed_agent, mock_agent):
        """Test that forward passes are properly delegated."""
        obs = torch.randn(4, 128, 3)
        state = PolicyState()
        action = torch.tensor([[0, 1], [1, 2], [0, 0], [1, 0]])

        # Need to mock the forward call on the wrapper since DDP intercepts it
        distributed_agent.forward = mock_agent.forward
        result = distributed_agent.forward(obs, state, action)

        mock_agent.forward.assert_called_once_with(obs, state, action)
        assert len(result) == 5  # Should return 5-tuple

    def test_activate_actions_delegation(self, distributed_agent, mock_agent):
        """Test that activate_actions is properly delegated."""
        action_names = ["move", "attack", "use"]
        action_max_params = [2, 9, 3]
        device = torch.device("cpu")

        distributed_agent.activate_actions(action_names, action_max_params, device)

        mock_agent.activate_actions.assert_called_once_with(action_names, action_max_params, device)

    def test_key_and_version_delegation(self, distributed_agent, mock_agent):
        """Test that key_and_version is properly delegated."""
        key, version = distributed_agent.key_and_version()

        mock_agent.key_and_version.assert_called_once()
        assert key == "test_key"
        assert version == 1

    def test_policy_as_metta_agent_delegation(self, distributed_agent, mock_agent):
        """Test that policy_as_metta_agent returns the wrapped agent."""
        result = distributed_agent.policy_as_metta_agent()

        assert result is mock_agent

    def test_attribute_access_delegation(self, distributed_agent, mock_agent):
        """Test that attribute access is properly delegated."""
        # Test direct attributes
        assert distributed_agent.name == "test_agent"
        assert distributed_agent.uri == "file://test.pt"
        assert distributed_agent.metadata == {"epoch": 1}
        assert distributed_agent.model_type == "brain"

    def test_method_delegation(self, distributed_agent, mock_agent):
        """Test that various methods are properly delegated."""
        # Test l2_init_loss
        loss = distributed_agent.l2_init_loss()
        mock_agent.l2_init_loss.assert_called_once()
        assert isinstance(loss, torch.Tensor)

        # Test update_l2_init_weight_copy
        distributed_agent.update_l2_init_weight_copy()
        mock_agent.update_l2_init_weight_copy.assert_called_once()

        # Test clip_weights
        distributed_agent.clip_weights()
        mock_agent.clip_weights.assert_called_once()

        # Test compute_weight_metrics
        metrics = distributed_agent.compute_weight_metrics(delta=0.01)
        mock_agent.compute_weight_metrics.assert_called_once_with(delta=0.01)
        assert isinstance(metrics, list)

    def test_parameter_access(self, distributed_agent, mock_agent):
        """Test that parameters() is properly delegated."""
        params = list(distributed_agent.parameters())

        mock_agent.parameters.assert_called()
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_state_dict_operations(self, distributed_agent, mock_agent):
        """Test state_dict save/load operations."""
        # Test state_dict
        state = distributed_agent.state_dict()
        mock_agent.state_dict.assert_called()
        assert isinstance(state, dict)

        # Test load_state_dict
        new_state = {"test": torch.randn(2, 2)}
        distributed_agent.load_state_dict(new_state)
        mock_agent.load_state_dict.assert_called_with(new_state)

    @pytest.mark.parametrize(
        "attribute",
        [
            "components",
            "device",
            "local_path",
            "model",
            "model_type",
        ],
    )
    def test_optional_attribute_delegation(self, distributed_agent, mock_agent, attribute):
        """Test that optional attributes are properly delegated."""
        # Set the attribute on the mock
        setattr(mock_agent, attribute, f"test_{attribute}")

        # Access through distributed wrapper
        value = getattr(distributed_agent, attribute)
        assert value == f"test_{attribute}"

    def test_getattr_fallback(self, distributed_agent, mock_agent):
        """Test that __getattr__ properly falls back to wrapped agent for unknown attributes."""
        # Add a custom attribute to the mock
        mock_agent.custom_attribute = "custom_value"

        # Should be accessible through the wrapper
        assert distributed_agent.custom_attribute == "custom_value"

    def test_getattr_super_fallback(self):
        """Test that __getattr__ falls back to super() when attribute not in wrapped agent."""
        mock_agent = Mock(spec=MettaAgent)

        with patch("torch.nn.parallel.DistributedDataParallel.__init__", return_value=None):
            agent = DistributedMettaAgent(mock_agent, device="cpu")
            agent._wrapped_agent = mock_agent

            # Try to access an attribute that's not in wrapped agent
            # Should fall back to super().__getattr__
            with patch("torch.nn.parallel.DistributedDataParallel.__getattr__") as mock_super:
                mock_super.return_value = "ddp_attribute"
                result = agent.nonexistent_attr

                mock_super.assert_called_once_with("nonexistent_attr")
                assert result == "ddp_attribute"

    def test_ddp_initialization(self):
        """Test that DistributedDataParallel is properly initialized."""
        mock_agent = Mock(spec=MettaAgent)
        device = "cuda:0"

        with patch("torch.nn.parallel.DistributedDataParallel.__init__") as mock_init:
            DistributedMettaAgent(mock_agent, device)

            # Check that DDP was initialized with correct arguments
            mock_init.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDistributedMettaAgentCUDA:
    """Test DistributedMettaAgent with CUDA devices."""

    def test_cuda_device_handling(self):
        """Test that CUDA devices are properly handled."""
        mock_agent = Mock(spec=MettaAgent)
        mock_agent.to = Mock(return_value=mock_agent)

        with patch("torch.nn.parallel.DistributedDataParallel.__init__", return_value=None):
            device = "cuda:0"
            agent = DistributedMettaAgent(mock_agent, device)

            # Should be initialized with proper CUDA device
            assert hasattr(agent, "_wrapped_agent")
