from __future__ import annotations

import random
import numpy as np
import pytest

from experiments.recipes.assembly_lines import (
    AssemblyLinesTaskGenerator,
    curriculum_args,
    make_task_generator_cfg,
    size_ranges,
)
from metta.cogworks.curriculum.config_serialization import (
    deserialize_config,
    extract_features_from_config,
    get_feature_dim,
    get_feature_spec,
    serialize_config,
)


@pytest.fixture
def task_generator():
    cfg = make_task_generator_cfg(**curriculum_args["full"])
    return AssemblyLinesTaskGenerator(cfg)


def _flat_from_blocks(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([d["continuous"], d["categorical"]], axis=0)


def test_feature_spec_and_dim():
    spec = get_feature_spec()
    assert len(spec) > 0
    assert all(s.feature_type in {"continuous", "discrete", "categorical"} for s in spec)
    dim = get_feature_dim()
    assert 10 <= dim <= 30


def test_extract_features_structured_first(task_generator):
    cfg = task_generator.get_task(42)
    feats = extract_features_from_config(cfg)

    assert "width" in feats and 5 <= feats["width"] <= 50
    assert "height" in feats and 5 <= feats["height"] <= 50
    assert "chain_length" in feats and 1 <= feats["chain_length"] <= 6
    assert "num_sinks" in feats and 0 <= feats["num_sinks"] <= 2
    # keeps xlarge in the acceptable set (future use)
    assert feats["room_size"] in {"tiny", "small", "medium", "large", "xlarge"}
    assert feats["terrain"] in {"no-terrain", "sparse", "balanced", "dense"}


def test_serialize_base_and_dict_shapes(task_generator):
    cfg = task_generator.get_task(42)

    d = serialize_config(cfg)
    vec = _flat_from_blocks(d)

    assert vec.dtype == np.float32
    assert vec.shape == (get_feature_dim(),)
    assert np.all(np.isfinite(vec))
    assert np.all(vec >= -1e-4) and np.all(vec <= 1 + 1e-4)

    assert isinstance(d["continuous"], np.ndarray)
    assert isinstance(d["categorical"], np.ndarray)
    assert d["continuous"].ndim == 1 and d["categorical"].ndim == 1
    assert d["continuous"].size + d["categorical"].size == get_feature_dim()


def test_roundtrip(task_generator):
    """Serialize -> Deserialize should preserve base fields exactly."""
    cfg0 = task_generator.get_task(42)

    base = serialize_config(cfg0)
    cfg1 = deserialize_config(base)

    f0 = extract_features_from_config(cfg0)
    f1 = extract_features_from_config(cfg1)

    assert (f0["width"], f0["height"]) == (f1["width"], f1["height"])
    assert (f0["chain_length"], f0["num_sinks"]) == (f1["chain_length"], f1["num_sinks"])
    assert f0["terrain"] == f1["terrain"]
    assert f0["room_size"] == f1["room_size"]
    assert cfg0.label == cfg1.label


def test_multiple_configs(task_generator):
    """Serialization on diverse configs is stable and finite."""
    for task_id in range(50):
        cfg = task_generator.get_task(task_id)
        d = serialize_config(cfg)
        vec = _flat_from_blocks(d)
        assert vec.shape == (get_feature_dim(),)
        assert np.all(np.isfinite(vec))


def test_determinism(task_generator):
    cfg = task_generator.get_task(99)
    a = serialize_config(cfg)
    b = serialize_config(cfg)
    np.testing.assert_array_equal(a["continuous"], b["continuous"])
    np.testing.assert_array_equal(a["categorical"], b["categorical"])


def _cat_lengths_from_spec():
    """Return (len_room_size, len_terrain) from spec."""
    rs = next(s for s in get_feature_spec() if s.name == "room_size")
    tr = next(s for s in get_feature_spec() if s.name == "terrain")
    return len(rs.categories or []), len(tr.categories or [])


@pytest.mark.parametrize("room_size", ["tiny", "small", "medium", "large"])
def test_min_max_edges_roundtrip(task_generator, room_size):
    """Edge-case dims at min/max per room_size round-trip exactly."""
    lo, hi = size_ranges[room_size]
    rng = random.Random(1)
    for w, h in [(lo, lo), (hi, hi)]:
        cfg = task_generator.build_config_from_params(
            chain_length=6,
            num_sinks=2,
            width=w,
            height=h,
            terrain="dense",
            room_size=room_size,
            rng=rng,
        )
        rt = deserialize_config(serialize_config(cfg))
        f0 = extract_features_from_config(cfg)
        f1 = extract_features_from_config(rt)
        assert (f0["width"], f0["height"]) == (f1["width"], f1["height"])
        assert f0["room_size"] == f1["room_size"]
        assert f0["terrain"] == f1["terrain"]
        assert (f0["chain_length"], f0["num_sinks"]) == (f1["chain_length"], f1["num_sinks"])


def test_categorical_segments_are_one_hot(task_generator):
    """Room size and terrain segments are proper one-hots."""
    d = serialize_config(task_generator.get_task(11))
    n_rs, n_tr = _cat_lengths_from_spec()
    rs = d["categorical"][0:n_rs]
    tr = d["categorical"][n_rs : n_rs + n_tr]
    assert np.isclose(rs.sum(), 1.0, atol=1e-6)
    assert np.isclose(tr.sum(), 1.0, atol=1e-6)
    assert (rs >= -1e-6).all() and (tr >= -1e-6).all()


@pytest.mark.parametrize("terrain", ["no-terrain", "sparse", "balanced", "dense"])
def test_all_terrains_roundtrip(task_generator, terrain):
    """Ensure each terrain survives serialize/deserialize."""
    cfg = task_generator.build_config_from_params(
        chain_length=3,
        num_sinks=1,
        width=12,
        height=9,
        terrain=terrain,
        room_size="small",
        rng=random.Random(0),
    )
    rt = deserialize_config(serialize_config(cfg))
    f0, f1 = extract_features_from_config(cfg), extract_features_from_config(rt)
    assert f0["terrain"] == f1["terrain"]


def test_assemblers_block_differs_when_structure_differs(task_generator):
    """Optional assembler block should reflect structure (even with count-only fallback)."""
    a = task_generator.build_config_from_params(
        1,
        0,
        10,
        10,
        "sparse",
        "small",
        random.Random(0),
    )
    b = task_generator.build_config_from_params(
        6,
        2,
        10,
        10,
        "sparse",
        "small",
        random.Random(0),
    )
    sa = serialize_config(a, include_assemblers=True)
    sb = serialize_config(b, include_assemblers=True)
    assert "assemblers" in sa and "assemblers" in sb
    assert sa["assemblers"].shape == sb["assemblers"].shape
    assert not np.allclose(sa["assemblers"], sb["assemblers"])
