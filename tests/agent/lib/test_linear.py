import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import Linear


@pytest.fixture
def simple_linear_environment():
    """Create a minimal environment for testing the Linear layer."""
    # Define the dimensions
    batch_size = 4
    input_size = 10
    output_size = 16

    # Create input data
    sample_input = {
        "input_tensor": torch.rand(batch_size, input_size),
    }

    # Create Linear layer
    # Note: Linear expects _nn_params as an OmegaConf object
    linear_layer = Linear(
        name="_linear_test_",
        sources=[{"name": "input_tensor"}],
        nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
        nonlinearity=None,
        initialization="Orthogonal",
        clip_scale=1.0,
        l2_norm_scale=0.01,
        l2_init_scale=0.005,
    )

    # Set up in_tensor_shapes manually
    linear_layer._in_tensor_shapes = [[input_size]]

    # Initialize the network
    if not hasattr(linear_layer, "_out_tensor_shape"):
        linear_layer._out_tensor_shape = [output_size]

    # Set up the network manually
    linear_layer._initialize()

    # Return all components needed for testing
    return {
        "linear_layer": linear_layer,
        "sample_input": sample_input,
        "params": {
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
        },
    }


class TestLinearLayer:
    """Tests for the Linear layer focusing on forward pass, initialization, and regularization."""

    def test_basic_forward_pass(self, simple_linear_environment):
        """Test basic forward pass of the Linear layer."""
        linear_layer = simple_linear_environment["linear_layer"]
        sample_input = simple_linear_environment["sample_input"]
        params = simple_linear_environment["params"]

        # Create tensordict with input
        td = {"input_tensor": sample_input["input_tensor"]}

        # Forward pass
        result = linear_layer._forward(td)

        # Verify output shape
        assert result[linear_layer._name].shape == (params["batch_size"], params["output_size"])

        # Verify output is not all zeros
        assert not torch.allclose(result[linear_layer._name], torch.zeros_like(result[linear_layer._name]))

    def test_repeated_forward_pass(self, simple_linear_environment):
        """Test that repeated forward passes with the same input produce the same output."""
        linear_layer = simple_linear_environment["linear_layer"]
        sample_input = simple_linear_environment["sample_input"]

        # First forward pass
        td1 = {"input_tensor": sample_input["input_tensor"]}
        result1 = linear_layer._forward(td1)
        output1 = result1[linear_layer._name]

        # Second forward pass with same input
        td2 = {"input_tensor": sample_input["input_tensor"]}
        result2 = linear_layer._forward(td2)
        output2 = result2[linear_layer._name]

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_different_inputs(self, simple_linear_environment):
        """Test that different inputs produce different outputs."""
        linear_layer = simple_linear_environment["linear_layer"]
        sample_input = simple_linear_environment["sample_input"]
        params = simple_linear_environment["params"]

        # First forward pass
        td1 = {"input_tensor": sample_input["input_tensor"]}
        result1 = linear_layer._forward(td1)
        output1 = result1[linear_layer._name]

        # Second forward pass with different input
        different_input = torch.rand(params["batch_size"], params["input_size"])
        td2 = {"input_tensor": different_input}
        result2 = linear_layer._forward(td2)
        output2 = result2[linear_layer._name]

        # Outputs should be different
        assert not torch.allclose(output1, output2)

    def test_weight_initialization(self, simple_linear_environment):
        """Test the weight initialization of the Linear layer."""
        linear_layer = simple_linear_environment["linear_layer"]

        # Check that weights are initialized (not all zeros)
        assert not torch.allclose(
            linear_layer.weight_net.weight.data, torch.zeros_like(linear_layer.weight_net.weight.data)
        )

        # For orthogonal initialization, test if weight matrix is close to orthogonal
        # For a tall matrix W (more rows than columns), W.T @ W should be close to identity
        # For a fat matrix W (more columns than rows), W @ W.T should be close to identity
        W = linear_layer.weight_net.weight.data

        if W.shape[0] <= W.shape[1]:  # If output_size <= input_size
            # W @ W.T should be close to identity
            product = torch.mm(W, W.t())
            identity = torch.eye(W.shape[0], device=W.device)
            # Allow some tolerance due to numerical issues
            close_to_orthogonal = torch.allclose(product, identity, atol=1e-6)
            assert close_to_orthogonal, "Weight initialization should be orthogonal"
        else:
            # W.T @ W should be close to identity
            product = torch.mm(W.t(), W)
            identity = torch.eye(W.shape[1], device=W.device)
            # Allow some tolerance
            close_to_orthogonal = torch.allclose(product, identity, atol=1e-6)
            assert close_to_orthogonal, "Weight initialization should be orthogonal"

        # Check bias initialization (should be zeros)
        assert torch.allclose(linear_layer.weight_net.bias.data, torch.zeros_like(linear_layer.weight_net.bias.data))

    def test_weight_clipping(self, simple_linear_environment):
        """Test the weight clipping functionality."""
        linear_layer = simple_linear_environment["linear_layer"]

        # Store original weights
        original_weights = linear_layer.weight_net.weight.data.clone()

        # Modify weights to have values outside clipping range
        with torch.no_grad():
            # Set some weights to large values (10x the clipping value)
            large_value = linear_layer.clip_value * 10
            linear_layer.weight_net.weight.data[0, 0] = large_value
            linear_layer.weight_net.weight.data[1, 1] = -large_value

        # Apply weight clipping
        linear_layer.clip_weights()

        # Check that large weights have been clipped
        assert abs(linear_layer.weight_net.weight.data[0, 0]) <= linear_layer.clip_value
        assert abs(linear_layer.weight_net.weight.data[1, 1]) <= linear_layer.clip_value

        # Check that other weights haven't changed significantly
        modified_weights = torch.ones_like(original_weights, dtype=torch.bool)
        modified_weights[0, 0] = False
        modified_weights[1, 1] = False

        assert torch.allclose(linear_layer.weight_net.weight.data[modified_weights], original_weights[modified_weights])

    def test_l2_regularization(self, simple_linear_environment):
        """Test the L2 regularization loss calculation."""
        linear_layer = simple_linear_environment["linear_layer"]

        # Calculate L2 regularization loss
        l2_loss = linear_layer.l2_reg_loss()

        # Verify loss is non-negative
        assert l2_loss >= 0

        # Calculate expected loss manually
        expected_l2_loss = torch.sum(linear_layer.weight_net.weight.data**2) * linear_layer.l2_norm_scale

        # Verify loss calculation is correct
        assert torch.allclose(l2_loss, expected_l2_loss)

        # Test with different l2_norm_scale
        original_scale = linear_layer.l2_norm_scale
        linear_layer.l2_norm_scale = original_scale * 2

        # Loss should be proportional to scale
        new_l2_loss = linear_layer.l2_reg_loss()
        assert torch.allclose(new_l2_loss, expected_l2_loss * 2)

        # Reset scale
        linear_layer.l2_norm_scale = original_scale

    def test_l2_init_regularization(self, simple_linear_environment):
        """Test the L2-init regularization loss calculation."""
        linear_layer = simple_linear_environment["linear_layer"]

        # Initial L2-init loss should be zero (weights haven't changed from initial values)
        initial_l2_init_loss = linear_layer.l2_init_loss()
        assert torch.allclose(initial_l2_init_loss, torch.tensor(0.0, device=initial_l2_init_loss.device))

        # Modify weights
        with torch.no_grad():
            # Change weights by a small amount
            delta = 0.1
            linear_layer.weight_net.weight.data += delta

        # Calculate L2-init loss after modification
        l2_init_loss = linear_layer.l2_init_loss()

        # Calculate expected loss manually
        expected_l2_init_loss = (
            torch.sum(delta**2 * torch.ones_like(linear_layer.weight_net.weight.data)) * linear_layer.l2_init_scale
        )

        # Verify loss calculation is correct
        assert torch.allclose(l2_init_loss, expected_l2_init_loss)

    def test_update_l2_init_weight_copy(self, simple_linear_environment):
        """Test updating the initial weight reference for L2-init regularization."""
        linear_layer = simple_linear_environment["linear_layer"]

        # Store initial weights
        initial_weights = linear_layer.initial_weights.clone()

        # Modify current weights
        with torch.no_grad():
            delta = 0.1
            linear_layer.weight_net.weight.data += delta

        # Update initial weights reference with alpha = 0.5
        alpha = 0.5
        linear_layer.update_l2_init_weight_copy(alpha)

        # Expected updated initial weights
        expected_updated_weights = alpha * initial_weights + (1 - alpha) * linear_layer.weight_net.weight.data

        # Verify updated initial weights
        assert torch.allclose(linear_layer.initial_weights, expected_updated_weights)

    def test_nonlinearity(self):
        """Test the Linear layer with different non-linearities."""
        input_size = 5  # Make this match the test input dimension
        output_size = 16

        # Test with ReLU
        linear_relu = Linear(
            name="_linear_relu_",
            sources=[{"name": "input_tensor"}],
            nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
            nonlinearity="nn.ReLU",
        )
        linear_relu._in_tensor_shapes = [[input_size]]  # Match the input size to the test data
        linear_relu._initialize()

        # Verify that _net is a Sequential with Linear and ReLU
        assert isinstance(linear_relu._net, torch.nn.Sequential)
        assert isinstance(linear_relu._net[0], torch.nn.Linear)
        assert isinstance(linear_relu._net[1], torch.nn.ReLU)

        # Test with Tanh
        linear_tanh = Linear(
            name="_linear_tanh_",
            sources=[{"name": "input_tensor"}],
            nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
            nonlinearity="nn.Tanh",
        )
        linear_tanh._in_tensor_shapes = [[input_size]]  # Match the input size to the test data
        linear_tanh._initialize()

        # Verify that _net is a Sequential with Linear and Tanh
        assert isinstance(linear_tanh._net, torch.nn.Sequential)
        assert isinstance(linear_tanh._net[0], torch.nn.Linear)
        assert isinstance(linear_tanh._net[1], torch.nn.Tanh)

        # Test behavior with negative inputs for ReLU
        input_tensor = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])  # 5 elements
        td = {"input_tensor": input_tensor}

        # Forward pass through ReLU layer
        result_relu = linear_relu._forward(td)

        # ReLU should eliminate negative values in output
        assert (result_relu[linear_relu._name] >= 0).all()

    def test_different_initializations(self):
        """Test different weight initialization methods."""
        input_size = 10
        output_size = 16

        # Test Xavier initialization
        linear_xavier = Linear(
            name="_linear_xavier_",
            sources=[{"name": "input_tensor"}],
            nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
            initialization="Xavier",
        )
        linear_xavier._in_tensor_shapes = [[input_size]]
        linear_xavier._initialize()

        # Xavier weights should have specific standard deviation
        xavier_scale = np.sqrt(6.0 / (input_size + output_size))
        xavier_weights = linear_xavier.weight_net.weight.data
        assert abs(xavier_weights.std().item() - xavier_scale / np.sqrt(3)) < 0.1

        # Test Normal initialization
        linear_normal = Linear(
            name="_linear_normal_",
            sources=[{"name": "input_tensor"}],
            nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
            initialization="Normal",
        )
        linear_normal._in_tensor_shapes = [[input_size]]
        linear_normal._initialize()

        # Normal weights should have specific standard deviation
        normal_scale = np.sqrt(2.0 / input_size)
        normal_weights = linear_normal.weight_net.weight.data
        assert abs(normal_weights.std().item() - normal_scale) < 0.1

        # Test max_0_01 initialization
        linear_max_0_01 = Linear(
            name="_linear_max_0_01_",
            sources=[{"name": "input_tensor"}],
            nn_params=OmegaConf.create({"out_features": output_size, "bias": True}),
            initialization="max_0_01",
        )
        linear_max_0_01._in_tensor_shapes = [[input_size]]
        linear_max_0_01._initialize()

        # max_0_01 weights should have max abs value of 0.01
        max_0_01_weights = linear_max_0_01.weight_net.weight.data
        assert torch.max(torch.abs(max_0_01_weights)).item() <= 0.01

    # Fix for test_forward_with_tensordict
    def test_forward_with_tensordict(self, simple_linear_environment):
        """Test the forward method with TensorDict as input."""
        linear_layer = simple_linear_environment["linear_layer"]
        sample_input = simple_linear_environment["sample_input"]
        params = simple_linear_environment["params"]

        # Create tensordict
        batch_size = params["batch_size"]
        td = TensorDict({"input_tensor": sample_input["input_tensor"]}, batch_size=batch_size)

        # Mock the source components lookup if needed
        # This ensures _source_components exists and doesn't get accessed during forward
        if not hasattr(linear_layer, "_source_components"):
            linear_layer._source_components = None

        # Forward pass
        result_td = linear_layer.forward(td)

        # Verify output exists and has correct shape
        assert linear_layer._name in result_td
        assert result_td[linear_layer._name].shape == (batch_size, params["output_size"])

    def test_forward_with_cache(self, simple_linear_environment):
        """Test that forward method doesn't recompute if result already exists."""
        linear_layer = simple_linear_environment["linear_layer"]
        sample_input = simple_linear_environment["sample_input"]
        params = simple_linear_environment["params"]

        # Pre-compute result
        batch_size = params["batch_size"]
        output_size = params["output_size"]
        cached_result = torch.ones(batch_size, output_size)

        # Create tensordict with input and cached result
        td = TensorDict(
            {"input_tensor": sample_input["input_tensor"], linear_layer._name: cached_result}, batch_size=batch_size
        )

        # Forward pass should not modify the cached result
        result_td = linear_layer.forward(td)

        # Verify cached result was preserved
        assert torch.allclose(result_td[linear_layer._name], cached_result)
