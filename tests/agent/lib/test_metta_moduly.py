import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_moduly import LinearModule, MettaModule


def test_metta_module_init():
    m = MettaModule(in_keys=["a"], out_keys=["b"], input_shapes={"a": (3,)})
    assert m.in_keys == ["a"]
    assert m.out_keys == ["b"]
    assert m.input_shapes["a"] == (3,)
    assert m.output_shapes == {}


def test_metta_module_validate_shapes_pass():
    m = MettaModule(in_keys=["a"], input_shapes={"a": (2,)})
    td = TensorDict({"a": torch.randn(4, 2)}, batch_size=4)
    m.validate_shapes(td)  # Should not raise


def test_metta_module_validate_shapes_fail():
    m = MettaModule(in_keys=["a"], input_shapes={"a": (2,)})
    td = TensorDict({"a": torch.randn(4, 3)}, batch_size=4)
    with pytest.raises(ValueError, match="Input shape mismatch for 'a'"):
        m.validate_shapes(td)


def test_linear_module_forward_success():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (5, 2)


def test_linear_module_forward_shape_mismatch():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(5, 4)}, batch_size=5)
    with pytest.raises(ValueError, match="Input shape mismatch for 'x'"):
        module(td)


def test_linear_module_forward_missing_key():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({}, batch_size=5)
    with pytest.raises(KeyError):
        module(td)


# New tests for edge cases and additional scenarios


def test_metta_module_empty_keys():
    m = MettaModule(in_keys=[], out_keys=[])
    assert m.in_keys == []
    assert m.out_keys == []


def test_metta_module_none_keys():
    m = MettaModule(in_keys=None, out_keys=None)
    assert m.in_keys == []
    assert m.out_keys == []


def test_linear_module_batch_size_one():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(1, 3)}, batch_size=1)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (1, 2)


def test_linear_module_large_batch_size():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.randn(100, 3)}, batch_size=100)
    out_td = module(td)
    assert "y" in out_td
    assert out_td["y"].shape == (100, 2)


def test_linear_module_custom_keys():
    module = LinearModule(in_features=3, out_features=2, in_key="custom_in", out_key="custom_out")
    td = TensorDict({"custom_in": torch.randn(5, 3)}, batch_size=5)
    out_td = module(td)
    assert "custom_out" in out_td
    assert out_td["custom_out"].shape == (5, 2)


def test_linear_module_non_tensor_input():
    module = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    td = TensorDict({"x": torch.tensor([1, 2, 3], dtype=torch.int64)}, batch_size=5)  # Use an incompatible tensor type
    with pytest.raises(TypeError):
        module(td)


def test_linear_module_integration():
    module1 = LinearModule(in_features=3, out_features=2, in_key="x", out_key="y")
    module2 = LinearModule(in_features=2, out_features=1, in_key="y", out_key="z")
    td = TensorDict({"x": torch.randn(5, 3)}, batch_size=5)
    out_td = module1(td)
    out_td = module2(out_td)
    assert "z" in out_td
    assert out_td["z"].shape == (5, 1)
