"""Unit tests for specific MettaModules.

Author: Axel
Created: 2024-03-19
"""

import torch
from tensordict import TensorDict

from metta.agent.lib.metta_module import MettaLinear, MettaReLU


def test_metta_linear_initialization():
    """Test MettaLinear initialization."""
    module = MettaLinear(
        in_keys=["input"],
        out_keys=["output"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    assert module.in_keys == ["input"]
    assert module.out_keys == ["output"]
    assert module.linear.in_features == 2
    assert module.linear.out_features == 3


def test_metta_linear_forward():
    """Test MettaLinear forward pass."""
    module = MettaLinear(
        in_keys=["input"],
        out_keys=["output"],
        input_features_shape=[2],
        output_features_shape=[3],
    )
    # Initialize weights for deterministic output
    module.linear.weight.data = torch.ones_like(module.linear.weight)
    module.linear.bias.data = torch.zeros_like(module.linear.bias)

    td = TensorDict({"input": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    result = module(td)
    assert "output" in result
    assert result["output"].shape == (1, 3)
    assert torch.allclose(result["output"], torch.tensor([[3.0, 3.0, 3.0]]))


def test_metta_relu_initialization():
    """Test MettaReLU initialization."""
    module = MettaReLU(in_keys=["input"], out_keys=["output"])
    assert module.in_keys == ["input"]
    assert module.out_keys == ["output"]


def test_metta_relu_forward():
    """Test MettaReLU forward pass."""
    module = MettaReLU(in_keys=["input"], out_keys=["output"])
    td = TensorDict({"input": torch.tensor([-1.0, 0.0, 1.0])}, batch_size=[])
    result = module(td)
    assert "output" in result
    assert torch.allclose(result["output"], torch.tensor([0.0, 0.0, 1.0]))
