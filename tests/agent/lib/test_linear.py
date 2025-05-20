import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.lib.nn_layer_library import Linear


class TestLinearLayer:
    @pytest.fixture
    def mock_source_component(self):
        """Create a mock source component with a 1D output shape."""

        class MockSourceComponent:
            def __init__(self):
                self._out_tensor_shape = [8]
                self._name = "input_layer"
                self._ready = True

            def forward(self, td):
                if self._name not in td:
                    # Create random input tensor for testing
                    td[self._name] = torch.rand(td.batch_size, 8)
                return td

        return MockSourceComponent()

    def test_initialization(self):
        """Test that the Linear layer initializes correctly with proper attributes."""
        # Create a Linear layer
        layer = Linear(
            name="test_linear", sources=[{"name": "input_layer"}], nn_params={"out_features": 32, "bias": True}
        )

        # Verify initial state
        assert layer._name == "test_linear"
        assert layer._sources == [{"name": "input_layer"}]
        assert layer._nn_params["out_features"] == 32
        assert layer._nn_params["bias"] is True
        assert layer._ready is False
        assert layer._net is None

    def test_setup(self, mock_source_component):
        """Test the setup method which prepares the layer for use."""
        # Create a Linear layer
        layer = Linear(
            name="test_linear", sources=[{"name": "input_layer"}], nn_params={"out_features": 16, "bias": True}
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Verify the layer is ready
        assert layer._ready is True
        assert layer._out_tensor_shape == [16]  # Output shape should match out_features
        assert isinstance(layer._net, nn.Module)

    def test_forward(self, mock_source_component):
        """Test forward pass of the Linear layer."""
        # Create a Linear layer
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            nonlinearity=None,  # No nonlinearity for simpler testing
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Create input data
        batch_size = 4
        td = TensorDict({}, batch_size=batch_size)

        # Forward pass
        result_td = layer.forward(td)

        # Verify output shape and existence
        assert "test_linear" in result_td
        assert result_td["test_linear"].shape == torch.Size([batch_size, 16])

    def test_forward_with_existing_result(self, mock_source_component):
        """Test that forward pass doesn't recompute when result already exists."""
        # Create a Linear layer
        layer = Linear(
            name="test_linear", sources=[{"name": "input_layer"}], nn_params={"out_features": 16, "bias": True}
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Create input data with existing result
        batch_size = 4
        existing_result = torch.ones(batch_size, 16)  # Distinct from what computation would produce
        td = TensorDict(
            {"input_layer": torch.rand(batch_size, 8), "test_linear": existing_result}, batch_size=batch_size
        )

        # Forward pass
        result_td = layer.forward(td)

        # Verify the existing result was preserved
        assert torch.all(result_td["test_linear"] == existing_result)

    def test_forward_with_nonlinearity(self, mock_source_component):
        """Test the Linear layer with a nonlinearity applied."""
        # Create a Linear layer with ReLU nonlinearity
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            nonlinearity="nn.ReLU",
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Verify the layer has a Sequential network with a ReLU
        assert isinstance(layer._net, nn.Sequential)
        assert isinstance(layer._net[0], nn.Linear)
        assert isinstance(layer._net[1], nn.ReLU)

        # Create input with negative values to test ReLU
        batch_size = 4
        input_data = torch.randn(batch_size, 8)  # Will have both positive and negative values
        td = TensorDict({"input_layer": input_data}, batch_size=batch_size)

        # Forward pass
        result_td = layer.forward(td)

        # Verify output shape and that all values are non-negative (ReLU effect)
        assert "test_linear" in result_td
        assert result_td["test_linear"].shape == torch.Size([batch_size, 16])
        assert (result_td["test_linear"] >= 0).all()

    def test_invalid_input_shape(self):
        """Test that the layer raises an error with invalid input shape."""
        # Create a Linear layer
        layer = Linear(
            name="test_linear", sources=[{"name": "input_layer"}], nn_params={"out_features": 16, "bias": True}
        )

        # Create a mock source component with invalid shape (2D instead of 1D)
        class InvalidSourceComponent:
            def __init__(self):
                self._out_tensor_shape = [8, 4]  # 2D shape, should be 1D
                self._name = "input_layer"
                self._ready = True

            def forward(self, td):
                return td

        # Setup should raise an assertion error due to 2D input
        with pytest.raises(AssertionError):
            layer.setup({"input_layer": InvalidSourceComponent()})

    def test_weight_initialization_orthogonal(self, mock_source_component):
        """Test orthogonal weight initialization method."""
        # Create layer with orthogonal initialization
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            initialization="Orthogonal",
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Get weight matrix
        weight_matrix = layer.weight_net.weight.data

        # Check orthogonality properties
        # For a tall matrix, WW^T should be close to identity
        if weight_matrix.shape[0] <= weight_matrix.shape[1]:
            product = torch.mm(weight_matrix, weight_matrix.t())
            identity = torch.eye(weight_matrix.shape[0], device=product.device)
            assert torch.allclose(product, identity, atol=1e-6)

    def test_weight_initialization_xavier(self, mock_source_component):
        """Test Xavier weight initialization method."""
        # Create layer with Xavier initialization
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            initialization="Xavier",
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Get weight matrix
        weight_matrix = layer.weight_net.weight.data

        # Check variance (should be around 6/(in + out))
        input_size = 8  # From mock_source_component
        output_size = 16  # From nn_params
        expected_var = 6 / (input_size + output_size)
        # Use standard deviation squared to compare with variance
        actual_var = weight_matrix.var().item()
        assert abs(actual_var - expected_var / 3) < 0.1  # Allow some tolerance

    def test_weight_initialization_normal(self, mock_source_component):
        """Test Normal weight initialization method."""
        # Create layer with Normal initialization
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            initialization="Normal",
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Get weight matrix
        weight_matrix = layer.weight_net.weight.data

        # Check standard deviation (should be around sqrt(2/in))
        input_size = 8  # From mock_source_component
        expected_std = np.sqrt(2 / input_size)
        actual_std = weight_matrix.std().item()
        assert abs(actual_std - expected_std) < 0.1  # Allow some tolerance

    def test_weight_initialization_max_0_01(self, mock_source_component):
        """Test Max_0_01 custom weight initialization method."""
        # Create layer with Max_0_01 initialization
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            initialization="max_0_01",
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Get weight matrix
        weight_matrix = layer.weight_net.weight.data

        # Check max value (should be <= 0.01)
        assert torch.max(torch.abs(weight_matrix)).item() <= 0.01

    def test_weight_clipping(self, mock_source_component):
        """Test weight clipping to prevent exploding gradients."""
        # Create layer with weight clipping
        clip_scale = 1.0
        global_clip_range = 0.5
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            clip_scale=clip_scale,
            global_clip_range=global_clip_range,
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Set weights to large values
        with torch.no_grad():
            layer.weight_net.weight.data[0, 0] = 10.0
            layer.weight_net.weight.data[0, 1] = -10.0

        # Apply weight clipping
        layer.clip_weights()

        # Check that weights are clipped
        assert layer.weight_net.weight.data.max().item() <= layer.clip_value
        assert layer.weight_net.weight.data.min().item() >= -layer.clip_value

    def test_l2_regularization(self, mock_source_component):
        """Test L2 regularization (weight decay)."""
        # Create layer with L2 regularization
        l2_scale = 0.01
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            l2_norm_scale=l2_scale,
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Compute L2 loss
        l2_loss = layer.l2_reg_loss()

        # Check L2 loss calculation
        expected_l2_loss = l2_scale * torch.sum(layer.weight_net.weight.data**2)
        assert torch.isclose(l2_loss, expected_l2_loss)

    def test_l2_init_regularization(self, mock_source_component):
        """Test L2-init regularization (delta regularization)."""
        # Create layer with L2-init regularization
        l2_init_scale = 0.01
        layer = Linear(
            name="test_linear",
            sources=[{"name": "input_layer"}],
            nn_params={"out_features": 16, "bias": True},
            l2_init_scale=l2_init_scale,
            nonlinearity=None,
        )

        # Setup the layer
        layer.setup({"input_layer": mock_source_component})

        # Compute initial L2-init loss (should be 0)
        initial_l2_init_loss = layer.l2_init_loss()
        assert initial_l2_init_loss.item() == 0

        # Store initial weights
        initial_weights = layer.initial_weights.clone()

        # Modify weights
        with torch.no_grad():
            layer.weight_net.weight.data += 0.1

        # Compute L2-init loss after weight modification
        l2_init_loss = layer.l2_init_loss()
        expected_l2_init_loss = l2_init_scale * torch.sum((layer.weight_net.weight.data - initial_weights) ** 2)
        assert torch.isclose(l2_init_loss, expected_l2_init_loss)

        # Test update L2-init weight copy
        alpha = 0.9
        layer.update_l2_init_weight_copy(alpha)
        expected_updated_weights = alpha * initial_weights + (1 - alpha) * layer.weight_net.weight.data
        assert torch.allclose(layer.initial_weights, expected_updated_weights)
