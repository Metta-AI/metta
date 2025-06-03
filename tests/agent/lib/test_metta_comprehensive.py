import time

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.merge_layer import MettaCenterPixel, MettaConcatMerge
from metta.agent.lib.metta_module import MettaConv2d, MettaDict, MettaFlatten, MettaLayerNorm, MettaLinear, MettaReLU
from metta.agent.lib.obs_shaper import MettaObsShaper
from metta.agent.lib.observation_normalizer import OBS_NORMALIZATIONS, MettaObsNormalizer


# MettaLinear
@pytest.mark.parametrize("batch, in_f, out_f", [(2, 8, 4), (1, 3, 3)])
def test_metta_linear_forward(batch, in_f, out_f):
    x = torch.randn(batch, in_f)
    td = TensorDict({"input": x}, batch_size=[batch])
    module = MettaLinear(["input"], ["output"], [in_f], [out_f])
    out = module(td)
    assert "output" in out
    assert out["output"].shape == (batch, out_f)
    # Shape validation
    with pytest.raises(ValueError):
        module(TensorDict({"input": torch.randn(batch, in_f + 1)}, batch_size=[batch]))


# MettaReLU
@pytest.mark.parametrize("shape", [(2, 4), (3, 2)])
def test_metta_relu_forward(shape):
    x = torch.randn(*shape)
    td = TensorDict({"input": x}, batch_size=[shape[0]])
    module = MettaReLU(["input"], ["output"])
    out = module(td)
    assert "output" in out
    torch.testing.assert_close(out["output"], torch.relu(x))


# MettaConv2d
@pytest.mark.parametrize("batch, in_c, out_c, h, w, k", [(2, 3, 4, 8, 8, 3)])
def test_metta_conv2d_forward(batch, in_c, out_c, h, w, k):
    x = torch.randn(batch, in_c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    out_h = (h - k) + 1
    out_w = (w - k) + 1
    module = MettaConv2d(["input"], ["output"], [in_c, h, w], [out_c, out_h, out_w], out_c, k)
    out = module(td)
    assert "output" in out
    assert out["output"].shape == (batch, out_c, out_h, out_w)


# MettaFlatten
@pytest.mark.parametrize("batch, c, h, w", [(2, 3, 4, 5)])
def test_metta_flatten_forward(batch, c, h, w):
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    module = MettaFlatten(["input"], ["output"], [c, h, w], [c * h * w])
    out = module(td)
    assert "output" in out
    assert out["output"].shape == (batch, c * h * w)


# MettaLayerNorm
@pytest.mark.parametrize("batch, f", [(2, 5)])
def test_metta_layernorm_forward(batch, f):
    x = torch.randn(batch, f)
    td = TensorDict({"input": x}, batch_size=[batch])
    module = MettaLayerNorm(["input"], ["output"], [f], [f], normalized_shape=f)
    out = module(td)
    assert "output" in out
    assert out["output"].shape == (batch, f)


# MettaCenterPixel
@pytest.mark.parametrize("batch, c, h, w", [(2, 3, 5, 7)])
def test_metta_center_pixel_forward(batch, c, h, w):
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    module = MettaCenterPixel(["input"], ["output"], [c, h, w], [c])
    out = module(td)
    assert "output" in out
    assert out["output"].shape == (batch, c)


# MettaConcatMerge
@pytest.mark.parametrize("batch, c1, c2, h, w", [(2, 3, 4, 5, 6)])
def test_metta_concat_merge_forward(batch, c1, c2, h, w):
    x1 = torch.randn(batch, c1, h, w)
    x2 = torch.randn(batch, c2, h, w)
    td = TensorDict({"a": x1, "b": x2}, batch_size=[batch])
    module = MettaConcatMerge(
        in_keys=["a", "b"],
        out_keys=["out"],
        input_features_shape=[[c1, h, w], [c2, h, w]],
        output_features_shape=[c1 + c2, h, w],
        dim=1,
    )
    out = module(td)
    assert "out" in out
    assert out["out"].shape == (batch, c1 + c2, h, w)


# MettaObsShaper
@pytest.mark.parametrize("batch, h, w, c", [(2, 4, 5, 6)])
def test_metta_obs_shaper_forward(batch, h, w, c):
    x = torch.randn(batch, h, w, c)
    td = TensorDict({"x": x}, batch_size=[])
    module = MettaObsShaper(
        in_keys=["x"],
        out_keys=["out"],
        input_features_shape=[h, w, c],
        output_features_shape=[c, h, w],
        obs_shape=[h, w, c],
    )
    md = MettaDict(td, {})
    out = module(md)
    assert "out" in out.td
    assert out.td["out"].shape == (batch, c, h, w)
    # Metadata
    assert out.data["_batch_size_"] == batch
    assert out.data["_TT_"] == 1
    assert out.data["_BxTT_"] == batch


# MettaObsNormalizer
@pytest.mark.parametrize("batch, num_features", [(2, 5)])
def test_metta_obs_normalizer_forward(batch, num_features):
    grid_features = list(list(OBS_NORMALIZATIONS.keys())[:num_features])
    x = torch.randn(batch, num_features, 1, 1)
    td = TensorDict({"input": x}, batch_size=[])
    module = MettaObsNormalizer(
        in_keys=["input"],
        out_keys=["output"],
        input_features_shape=[num_features, 1, 1],
        output_features_shape=[num_features, 1, 1],
        grid_features=grid_features,
    )
    md = MettaDict(td, {})
    out = module(md)
    assert "output" in out.td
    assert out.td["output"].shape == (batch, num_features, 1, 1)


# INTEGRATION TESTS


def test_linear_relu_chain():
    batch, in_f, hidden_f, out_f = 2, 8, 4, 3
    x = torch.randn(batch, in_f)
    td = TensorDict({"input": x}, batch_size=[batch])
    linear = MettaLinear(["input"], ["hidden"], [in_f], [hidden_f])
    relu = MettaReLU(["hidden"], ["output"])
    td1 = linear(td)
    td2 = relu(td1)
    assert "output" in td2
    # Reference
    ref = torch.relu(linear.linear(x))
    torch.testing.assert_close(td2["output"], ref)


def test_conv2d_flatten_linear_chain():
    batch, in_c, h, w, out_f = 2, 3, 8, 8, 5
    x = torch.randn(batch, in_c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    conv = MettaConv2d(["input"], ["conv"], [in_c, h, w], [4, 8, 8], 4, 3, padding=1)
    flatten = MettaFlatten(["conv"], ["flat"], [4, 8, 8], [4 * 8 * 8])
    linear = MettaLinear(["flat"], ["output"], [4 * 8 * 8], [out_f])
    td1 = conv(td)
    td2 = flatten(td1)
    td3 = linear(td2)
    assert "output" in td3
    # Reference
    ref = conv.conv(x)
    ref_flat = ref.view(batch, -1)
    ref_out = linear.linear(ref_flat)
    torch.testing.assert_close(td3["output"], ref_out)


def test_obs_shaper_conv_centerpixel_chain():
    batch, h, w, c = 2, 5, 5, 3
    x = torch.randn(batch, h, w, c)
    td = TensorDict({"x": x}, batch_size=[])
    obs_shaper = MettaObsShaper(["x"], ["obs"], [h, w, c], [c, h, w], obs_shape=[h, w, c])
    conv = MettaConv2d(["obs"], ["conv"], [c, h, w], [4, h, w], 4, 1)
    center = MettaCenterPixel(["conv"], ["center"], [4, h, w], [4])
    md = MettaDict(td, {})
    md1 = obs_shaper(md)
    md2 = conv(md1)
    md3 = center(md2)
    assert "center" in md3.td
    assert md3.td["center"].shape == (batch, 4)
    # Metadata propagation
    assert md3.data["_batch_size_"] == batch


def test_concat_merge_linear_chain():
    batch, c1, c2, h, w, out_f = 2, 3, 4, 5, 6, 7
    x1 = torch.randn(batch, c1, h, w)
    x2 = torch.randn(batch, c2, h, w)
    td = TensorDict({"a": x1, "b": x2}, batch_size=[batch])
    concat = MettaConcatMerge(
        in_keys=["a", "b"],
        out_keys=["merged"],
        input_features_shape=[[c1, h, w], [c2, h, w]],
        output_features_shape=[c1 + c2, h, w],
        dim=1,
    )
    flatten = MettaFlatten(["merged"], ["flat"], [c1 + c2, h, w], [(c1 + c2) * h * w])
    linear = MettaLinear(["flat"], ["output"], [(c1 + c2) * h * w], [out_f])
    td1 = concat(td)
    td2 = flatten(td1)
    td3 = linear(td2)
    assert "output" in td3
    # Reference
    ref = torch.cat([x1, x2], dim=1).view(batch, -1)
    ref_out = linear.linear(ref)
    torch.testing.assert_close(td3["output"], ref_out)


# MIGRATION (PARITY) TESTS


def test_linear_parity():
    import torch.nn as nn

    batch, in_f, out_f = 2, 8, 4
    x = torch.randn(batch, in_f)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = nn.Linear(in_f, out_f)
    metta = MettaLinear(["input"], ["output"], [in_f], [out_f])
    metta.linear.load_state_dict(legacy.state_dict())
    out_legacy = legacy(x)
    out_metta = metta(td)["output"]
    torch.testing.assert_close(out_legacy, out_metta)


def test_relu_parity():
    import torch.nn as nn

    x = torch.randn(2, 4)
    td = TensorDict({"input": x}, batch_size=[2])
    legacy = nn.ReLU()
    metta = MettaReLU(["input"], ["output"])
    out_legacy = legacy(x)
    out_metta = metta(td)["output"]
    torch.testing.assert_close(out_legacy, out_metta)


def test_conv2d_parity():
    import torch.nn as nn

    batch, in_c, out_c, h, w, k = 2, 3, 4, 8, 8, 3
    x = torch.randn(batch, in_c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = nn.Conv2d(in_c, out_c, k, padding=1)
    metta = MettaConv2d(["input"], ["output"], [in_c, h, w], [out_c, h, w], out_c, k, padding=1)
    metta.conv.load_state_dict(legacy.state_dict())
    out_legacy = legacy(x)
    out_metta = metta(td)["output"]
    torch.testing.assert_close(out_legacy, out_metta)


def test_flatten_parity():
    import torch.nn as nn

    batch, c, h, w = 2, 3, 4, 5
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = nn.Flatten()
    metta = MettaFlatten(["input"], ["output"], [c, h, w], [c * h * w])
    out_legacy = legacy(x)
    out_metta = metta(td)["output"]
    torch.testing.assert_close(out_legacy, out_metta)


def test_layernorm_parity():
    import torch.nn as nn

    batch, f = 2, 5
    x = torch.randn(batch, f)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = nn.LayerNorm(f)
    metta = MettaLayerNorm(["input"], ["output"], [f], [f], normalized_shape=f)
    metta.ln.load_state_dict(legacy.state_dict())
    out_legacy = legacy(x)
    out_metta = metta(td)["output"]
    torch.testing.assert_close(out_legacy, out_metta)


def test_centerpixel_parity():
    from metta.agent.lib.merge_layer import CenterPixelLayer

    batch, c, h, w = 2, 3, 5, 7
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = CenterPixelLayer("output")
    legacy._sources = [{"name": "input"}]
    legacy._name = "output"
    legacy._ready = True
    td_legacy = legacy._forward(td.clone())
    metta = MettaCenterPixel(["input"], ["output"], [c, h, w], [c])
    out_metta = metta(td)["output"]
    torch.testing.assert_close(td_legacy["output"], out_metta)


def test_concatmerge_parity():
    from metta.agent.lib.merge_layer import ConcatMergeLayer

    batch, c1, c2, h, w = 2, 3, 4, 5, 6
    x1 = torch.randn(batch, c1, h, w)
    x2 = torch.randn(batch, c2, h, w)
    td = TensorDict({"a": x1, "b": x2}, batch_size=[batch])
    legacy = ConcatMergeLayer("output")
    legacy._sources = [{"name": "a", "dim": 1}, {"name": "b", "dim": 1}]
    legacy._name = "output"
    legacy.dims = [1, 1]
    legacy._ready = True
    legacy._in_tensor_shapes = [list(x1.shape[1:]), list(x2.shape[1:])]
    legacy.setup(_source_components={"a": legacy, "b": legacy})
    legacy._merge_dim = 1
    legacy._out_tensor_shape = list(td["a"].shape[1:])
    td_legacy = legacy._merge([x1, x2], td.clone())
    metta = MettaConcatMerge(
        in_keys=["a", "b"],
        out_keys=["output"],
        input_features_shape=[[c1, h, w], [c2, h, w]],
        output_features_shape=[c1 + c2, h, w],
        dim=1,
    )
    out_metta = metta(td)["output"]
    torch.testing.assert_close(td_legacy["output"], out_metta)


def test_obsshaper_parity():
    from metta.agent.lib.obs_shaper import ObsShaper

    batch, h, w, c = 2, 4, 5, 6
    x = torch.randn(batch, h, w, c)
    td = TensorDict({"x": x}, batch_size=[])
    legacy = ObsShaper([h, w, c])

    def legacy_forward_dict(self, d):
        x = d["x"]
        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            raise ValueError("Shape mismatch")
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor dimensionality")
        x = x.reshape(B * TT, *space_shape)
        x = x.float()
        from einops import rearrange

        x = rearrange(x, "b h w c -> b c h w")
        d[legacy._name] = x
        d["_TT_"] = TT
        d["_batch_size_"] = B
        d["_BxTT_"] = B * TT
        return d

    legacy._forward = legacy_forward_dict.__get__(legacy, ObsShaper)
    legacy._name = "out"
    td_legacy = legacy._forward(TensorDict({"x": x}, batch_size=[]))
    metta = MettaObsShaper(["x"], ["out"], [h, w, c], [c, h, w], obs_shape=[h, w, c])
    md = MettaDict(td, {})
    out_metta = metta(md)
    torch.testing.assert_close(td_legacy["out"], out_metta.td["out"])
    assert td_legacy["_TT_"] == out_metta.data["_TT_"]
    assert td_legacy["_batch_size_"] == out_metta.data["_batch_size_"]
    assert td_legacy["_BxTT_"] == out_metta.data["_BxTT_"]


def test_obsnormalizer_parity():
    from metta.agent.lib.observation_normalizer import ObservationNormalizer

    batch, num_features = 2, 5
    grid_features = list(list(OBS_NORMALIZATIONS.keys())[:num_features])
    x = torch.randn(batch, num_features, 1, 1)
    td = TensorDict({"input": x}, batch_size=[])
    legacy = ObservationNormalizer(grid_features)
    legacy._in_tensor_shapes = [[num_features, 1, 1]]
    legacy._sources = [{"name": "input"}]
    legacy._name = "output"
    legacy._initialize()
    td_legacy = legacy._forward(td.clone())
    metta = MettaObsNormalizer(
        in_keys=["input"],
        out_keys=["output"],
        input_features_shape=[num_features, 1, 1],
        output_features_shape=[num_features, 1, 1],
        grid_features=grid_features,
    )
    md = MettaDict(td, {})
    out_metta = metta(md)
    torch.testing.assert_close(td_legacy["output"], out_metta.td["output"])


# Full network migration test


def test_full_network_migration():
    # Legacy: Flatten -> Linear -> ReLU -> Linear
    import torch.nn as nn

    batch, c, h, w, hidden, out_f = 2, 3, 4, 5, 8, 2
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    legacy = nn.Sequential(
        nn.Flatten(),
        nn.Linear(c * h * w, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_f),
    )
    # Metta
    flatten = MettaFlatten(["input"], ["flat"], [c, h, w], [c * h * w])
    linear1 = MettaLinear(["flat"], ["hidden"], [c * h * w], [hidden])
    relu = MettaReLU(["hidden"], ["relu"])
    linear2 = MettaLinear(["relu"], ["output"], [hidden], [out_f])
    # Copy weights
    linear1.linear.load_state_dict(legacy[1].state_dict())
    linear2.linear.load_state_dict(legacy[3].state_dict())
    # Forward
    td1 = flatten(td)
    td2 = linear1(td1)
    td3 = relu(td2)
    td4 = linear2(td3)
    out_legacy = legacy(x)
    out_metta = td4["output"]
    torch.testing.assert_close(out_legacy, out_metta)


# PERFORMANCE BENCHMARKS


def test_performance_metta_vs_legacy_manual():
    import torch.nn as nn

    batch, c, h, w, hidden, out_f = 32, 8, 16, 16, 128, 10
    x = torch.randn(batch, c, h, w)
    td = TensorDict({"input": x}, batch_size=[batch])
    # Legacy network
    legacy = nn.Sequential(
        nn.Conv2d(c, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * h * w, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_f),
    )
    # Metta network
    conv = MettaConv2d(["input"], ["conv"], [c, h, w], [16, h, w], 16, 3, padding=1)
    relu1 = MettaReLU(["conv"], ["relu1"])
    flatten = MettaFlatten(["relu1"], ["flat"], [16, h, w], [16 * h * w])
    linear1 = MettaLinear(["flat"], ["hidden"], [16 * h * w], [hidden])
    relu2 = MettaReLU(["hidden"], ["relu2"])
    linear2 = MettaLinear(["relu2"], ["output"], [hidden], [out_f])
    # Copy weights
    conv.conv.load_state_dict(legacy[0].state_dict())
    linear1.linear.load_state_dict(legacy[3].state_dict())
    linear2.linear.load_state_dict(legacy[5].state_dict())
    # Warmup
    for _ in range(5):
        legacy(x)
        td1 = conv(td)
        td2 = relu1(td1)
        td3 = flatten(td2)
        td4 = linear1(td3)
        td5 = relu2(td4)
        td6 = linear2(td5)
        td6["output"]
    # Timing
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        legacy(x)
    t1 = time.perf_counter()
    legacy_time = (t1 - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        td1 = conv(td)
        td2 = relu1(td1)
        td3 = flatten(td2)
        td4 = linear1(td3)
        td5 = relu2(td4)
        td6 = linear2(td5)
        td6["output"]
    t1 = time.perf_counter()
    metta_time = (t1 - t0) / N
    print(f"Legacy avg time: {legacy_time * 1e6:.2f} us, Metta avg time: {metta_time * 1e6:.2f} us")
    assert metta_time < 2.0 * legacy_time
