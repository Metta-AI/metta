"""Unit tests for ModularNetwork.

Author: Axel
Created: 2024-03-19
"""

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_module import MettaLinear, MettaReLU
from metta.agent.lib.modular_network import ModularNetwork


def test_modular_network_initialization():
    """Test ModularNetwork initialization."""
    network = ModularNetwork()
    assert len(network.nodes) == 0
    assert len(network.out_key_to_node) == 0


def test_modular_network_add_component():
    """Test adding components to the network."""
    network = ModularNetwork()
    linear = MettaLinear(
        in_keys=["input"],
        out_keys=["hidden"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    relu = MettaReLU(in_keys=["hidden"], out_keys=["output"])

    network.add_component("linear", linear)
    network.add_component("relu", relu)

    assert "linear" in network.nodes
    assert "relu" in network.nodes
    assert network.out_key_to_node["hidden"] == "linear"
    assert network.out_key_to_node["output"] == "relu"


def test_modular_network_duplicate_component():
    """Test adding duplicate component names."""
    network = ModularNetwork()
    linear1 = MettaLinear(
        in_keys=["input"],
        out_keys=["hidden"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    linear2 = MettaLinear(
        in_keys=["hidden"],
        out_keys=["output"],
        input_features_shape=[3],
        output_features_shape=[1],
    )

    network.add_component("linear", linear1)
    with pytest.raises(ValueError):
        network.add_component("linear", linear2)


def test_modular_network_forward():
    """Test network forward pass."""
    network = ModularNetwork()
    linear = MettaLinear(
        in_keys=["input"],
        out_keys=["hidden"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    relu = MettaReLU(in_keys=["hidden"], out_keys=["output"])

    # Initialize weights for deterministic output
    linear.linear.weight.data = torch.ones_like(linear.linear.weight)
    linear.linear.bias.data = torch.zeros_like(linear.linear.bias)

    network.add_component("linear", linear)
    network.add_component("relu", relu)

    td = TensorDict({"input": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    result = network(td)

    assert "hidden" in result
    assert "output" in result
    assert torch.allclose(result["hidden"], torch.tensor([[3.0, 3.0, 3.0]]))
    assert torch.allclose(result["output"], torch.tensor([[3.0, 3.0, 3.0]]))


def test_modular_network_missing_input():
    """Test network with missing input."""
    network = ModularNetwork()
    linear = MettaLinear(
        in_keys=["input"],
        out_keys=["hidden"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    network.add_component("linear", linear)

    td = TensorDict({}, batch_size=[])
    with pytest.raises(KeyError):
        network(td)
