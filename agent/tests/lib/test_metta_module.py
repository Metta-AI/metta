"""Unit tests for MettaModule base class.

Author: Axel
Created: 2024-03-19
"""

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_module import MettaModule


class DummyModule(MettaModule):
    """A dummy module for testing the base class."""

    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        input_features_shape: list[int] | None = None,
        output_features_shape: list[int] | None = None,
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)

    def _compute(self, td: TensorDict) -> dict[str, torch.Tensor]:
        return {out_key: td[in_key] * 2 for in_key, out_key in zip(self.in_keys, self.out_keys, strict=False)}


def test_metta_module_initialization():
    """Test MettaModule initialization."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    assert module.in_keys == ["input"]
    assert module.out_keys == ["output"]


def test_metta_module_forward():
    """Test MettaModule forward pass."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    td = TensorDict({"input": torch.tensor([1.0, 2.0])}, batch_size=[])
    result = module(td)
    assert "output" in result
    assert torch.allclose(result["output"], torch.tensor([2.0, 4.0]))


def test_metta_module_missing_input():
    """Test MettaModule with missing input key."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    td = TensorDict({}, batch_size=[])
    with pytest.raises(KeyError):
        module(td)


def test_metta_module_shape_validation():
    """Test MettaModule shape validation."""
    module = DummyModule(
        in_keys=["input"],
        out_keys=["output"],
        input_features_shape=[2],
        output_features_shape=[2],
    )
    # Valid shape
    td = TensorDict({"input": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
    result = module(td)
    assert result["output"].shape == (1, 2)

    # Invalid shape
    td = TensorDict({"input": torch.tensor([1.0])}, batch_size=[])
    with pytest.raises(ValueError):
        module(td)
