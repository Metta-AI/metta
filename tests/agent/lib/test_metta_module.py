"""Unit tests for MettaModule base class.

Author: Axel
Created: 2024-03-19
"""

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_module import MettaData, MettaModule


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

    def _compute(self, md: MettaData) -> dict[str, torch.Tensor]:
        # Double the input tensor and add a flag to metadata
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=False):
            md.data[out_key] = {"flag": "processed"}
        return {out_key: md[in_key] * 2 for in_key, out_key in zip(self.in_keys, self.out_keys, strict=False)}


def test_metta_module_initialization():
    """Test MettaModule initialization."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    assert module.in_keys == ["input"]
    assert module.out_keys == ["output"]


def test_metta_module_forward():
    """Test MettaModule forward pass."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    td = TensorDict({"input": torch.tensor([1.0, 2.0])}, batch_size=[])
    md = MettaData(td, {})
    result = module(md)
    assert "output" in result
    assert torch.allclose(result["output"], torch.tensor([2.0, 4.0]))
    # Check metadata propagation
    assert "output" in result.data
    assert result.data["output"]["flag"] == "processed"


def test_metta_module_forward_tensordict():
    """Test MettaModule forward pass with TensorDict input (should return TensorDict)."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    td = TensorDict({"input": torch.tensor([1.0, 2.0])}, batch_size=[])
    result = module(td)
    assert isinstance(result, TensorDict)
    assert "output" in result
    assert torch.allclose(result["output"], torch.tensor([2.0, 4.0]))


def test_metta_module_missing_input():
    """Test MettaModule with missing input key."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    md = MettaData(TensorDict({}, batch_size=[]), {})
    with pytest.raises(KeyError):
        module(md)


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
    md = MettaData(td, {})
    result = module(md)
    assert result["output"].shape == (1, 2)

    # Invalid shape
    td = TensorDict({"input": torch.tensor([1.0])}, batch_size=[])
    md = MettaData(td, {})
    with pytest.raises(ValueError):
        module(md)


def test_metta_module_metadata_propagation():
    """Test that metadata is propagated and updated correctly."""
    module = DummyModule(in_keys=["input"], out_keys=["output"])
    td = TensorDict({"input": torch.tensor([1.0, 2.0])}, batch_size=[])
    md = MettaData(td, {"custom": "info"})
    result = module(md)
    assert result.data["custom"] == "info"
    assert "output" in result.data
    assert result.data["output"]["flag"] == "processed"
