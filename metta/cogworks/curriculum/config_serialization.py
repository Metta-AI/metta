from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mettagrid.config.mettagrid_config import MettaGridConfig


@dataclass
class FeatureSpec:
    """Specification for a feature dimension.

    For continuous/discrete types, values are normalized to [0, 1] with
    out-of-range values clipped before conversion.

    """

    name: str
    feature_type: str  # "continuous" | "discrete" | "categorical"
    min_value: float | None = None
    max_value: float | None = None
    categories: List[str] | None = None

    def normalize(self, value: float | int | str) -> float | np.ndarray:
        """Normalize value to model-friendly representation."""
        if self.feature_type in ("continuous", "discrete"):
            assert self.min_value is not None and self.max_value is not None, f"{self.name}: no min/max"
            rng = float(self.max_value - self.min_value)
            if rng <= 0:
                raise ValueError(f"{self.name}: invalid min/max")
            v = (float(value) - float(self.min_value)) / rng
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            return float(v)
        elif self.feature_type == "categorical":
            assert self.categories and len(self.categories) > 0, f"{self.name}: no categories"
            if not isinstance(value, str):
                value = str(value)
            if value not in self.categories:
                raise ValueError(f"{self.name}: unknown category '{value}'")
            one_hot: np.ndarray = np.zeros(len(self.categories), dtype=np.float32)
            one_hot[self.categories.index(value)] = 1.0
            return one_hot
        raise ValueError(f"Unknown feature type: {self.feature_type}")

    def denormalize(self, value: float | np.ndarray) -> float | int | str:
        """Convert normalized value back to original representation."""
        if self.feature_type in ("continuous", "discrete"):
            assert self.min_value is not None and self.max_value is not None
            raw = float(value) * (self.max_value - self.min_value) + self.min_value
            return int(round(raw)) if self.feature_type == "discrete" else float(raw)
        elif self.feature_type == "categorical":
            assert self.categories
            arr = np.asarray(value)
            idx = int(np.argmax(arr))
            return self.categories[idx]
        raise ValueError(f"Unknown feature type: {self.feature_type}")


_BASE_SPECS: List[FeatureSpec] = [
    FeatureSpec("width", "discrete", 5, 50),
    FeatureSpec("height", "discrete", 5, 50),
    FeatureSpec("chain_length", "discrete", 1, 6),
    FeatureSpec("num_sinks", "discrete", 0, 2),
    # Keeps 'xlarge' in the schema (may be used later by curricula)
    FeatureSpec("room_size", "categorical", categories=["tiny", "small", "medium", "large", "xlarge"]),
    FeatureSpec("terrain", "categorical", categories=["no-terrain", "sparse", "balanced", "dense"]),
]


def get_feature_spec() -> List[FeatureSpec]:
    """Return the base feature specification."""
    return list(_BASE_SPECS)


def get_feature_dim() -> int:
    """Return total dimensionality of the base feature vector."""
    dim = 0
    for s in _BASE_SPECS:
        dim += len(s.categories) if s.categories else 1
    return dim


def extract_features_from_config(config: MettaGridConfig) -> Dict[str, Any]:
    label_core = (config.label or "").split("|", 1)[0]
    parts = label_core.split("_")
    room_size = parts[0]
    chain_length = int(parts[1].replace("chain", ""))
    num_sinks = int(parts[2].replace("sinks", ""))
    terrain = parts[3] if len(parts) > 3 else "no-terrain"

    mb = config.game.map_builder
    mb_core = getattr(mb, "instance", mb)

    width = getattr(mb_core, "width", None)
    height = getattr(mb_core, "height", None)
    if width is None or height is None:
        raise ValueError("map_builder.width/height missing (builder may be incomplete).")

    objs = getattr(getattr(config, "game", None), "objects", None) or {}
    num_assemblers = sum(1 for o in objs.values() if getattr(o, "type", "") == "assembler")

    return {
        "width": int(width),
        "height": int(height),
        "chain_length": chain_length,
        "num_sinks": num_sinks,
        "room_size": room_size,
        "terrain": terrain,
        "num_assemblers": num_assemblers,
    }


def serialize_config(
    config: MettaGridConfig,
    *,
    include_assemblers: bool = False,
    resource_types: Sequence[str] | None = None,
    sentinels: Tuple[str, str] = ("nothing", "heart"),
) -> dict[str, np.ndarray]:
    """Serialize MettaGridConfig into:
        {
          "continuous": 1-D float32 array,
          "categorical": 1-D float32 array,
          ["assemblers"]: 1-D float32 array (optional)
        }
    Base blocks (reversible):
      - continuous: normalized scalars for non-categorical specs
      - categorical: concatenated one-hots for categorical specs

    Optional "assemblers" block (non-invertible):
      - If `resource_types` is provided:
          summary vector from `_summary_vector(...)`
      - Otherwise:
          single scalar = normalized assembler count
    """
    raw = extract_features_from_config(config)
    continuous_vals: List[float] = []
    categorical_vals: List[float] = []

    for spec in _BASE_SPECS:
        norm = spec.normalize(raw[spec.name])
        if isinstance(norm, np.ndarray):
            categorical_vals.extend(norm.tolist())
        else:
            continuous_vals.append(float(norm))

    out: dict[str, np.ndarray] = {
        "continuous": np.asarray(continuous_vals, dtype=np.float32),
        "categorical": np.asarray(categorical_vals, dtype=np.float32),
    }

    if include_assemblers:
        if resource_types:
            out["assemblers"] = _summary_vector(config, resource_types=resource_types, sentinels=sentinels).astype(
                np.float32
            )
        else:
            # Minimal signal: normalized assembler count (chain+1+sinks capped at 12)
            num_assemblers = float(raw.get("num_assemblers", 0))
            out["assemblers"] = np.asarray([min(12.0, max(0.0, num_assemblers)) / 12.0], dtype=np.float32)

    return out


def _denormalize_from_blocks(continuous: np.ndarray, categorical: np.ndarray) -> Dict[str, Any]:
    """Inverse of `serialize_config` (base blocks only)."""
    cont = np.asarray(continuous, dtype=np.float32).ravel()
    cat = np.asarray(categorical, dtype=np.float32).ravel()

    raw: Dict[str, Any] = {}
    ci = 0
    ki = 0

    for spec in _BASE_SPECS:
        if spec.feature_type == "categorical":
            n = len(spec.categories or [])
            if ki + n > cat.size:
                raise ValueError("categorical block too short for spec")
            raw[spec.name] = spec.denormalize(cat[ki : ki + n])
            ki += n
        else:
            if ci + 1 > cont.size:
                raise ValueError("continuous block too short for spec")
            raw[spec.name] = spec.denormalize(float(cont[ci]))
            ci += 1

    def _clip_int(name: str, lo: int, hi: int) -> int:
        v = int(raw[name])
        return int(np.clip(v, lo, hi))

    raw["width"] = _clip_int("width", 5, 50)
    raw["height"] = _clip_int("height", 5, 50)
    raw["chain_length"] = _clip_int("chain_length", 1, 6)
    raw["num_sinks"] = _clip_int("num_sinks", 0, 2)
    raw["terrain"] = str(raw["terrain"])
    raw["room_size"] = str(raw["room_size"])
    return raw


def deserialize_config(features: Mapping[str, np.ndarray], *, rng: random.Random | None = None) -> MettaGridConfig:
    """Deserialize base feature back into a MettaGridConfig.

    Expects:

      features["continuous"]: 1-D array of normalized scalars
      features["categorical"]: 1-D array of concatenated one-hots

    Any auxiliary keys (e.g. "assemblers") are ignored.
    """
    if "continuous" not in features or "categorical" not in features:
        raise KeyError("features must include 'continuous' and 'categorical'")
    cont = np.asarray(features["continuous"], dtype=np.float32)
    cat = np.asarray(features["categorical"], dtype=np.float32)

    if cont.ndim != 1 or cat.ndim != 1:
        raise ValueError("'continuous' and 'categorical' must be 1-D arrays")

    expected_cont = sum(1 for s in _BASE_SPECS if s.feature_type != "categorical")
    expected_cat = sum(len(s.categories or []) for s in _BASE_SPECS if s.feature_type == "categorical")
    if cont.size != expected_cont or cat.size != expected_cat:
        raise ValueError(
            f"feature length mismatch: expected cont={expected_cont}, cat={expected_cat}; "
            f"got cont={cont.size}, cat={cat.size}"
        )

    raw = _denormalize_from_blocks(cont, cat)
    rng = rng or random.Random(42)

    from experiments.recipes.assembly_lines import (
        AssemblyLinesTaskGenerator,
        curriculum_args,
        make_task_generator_cfg,
    )

    task_gen = AssemblyLinesTaskGenerator(make_task_generator_cfg(**curriculum_args["full"]))

    cfg = task_gen.build_config_from_params(
        chain_length=int(raw["chain_length"]),
        num_sinks=int(raw["num_sinks"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        terrain=str(raw["terrain"]),
        room_size=str(raw["room_size"]),
        rng=rng,
    )

    try:
        cfg.label = f"{raw['room_size']}_{raw['chain_length']}chain_{raw['num_sinks']}sinks_{raw['terrain']}"
    except Exception:
        pass
    return cfg


def _objects(cfg: MettaGridConfig) -> dict[str, Any]:
    """Return game objects mapping, handling both config- and game-scoped layouts."""
    return getattr(cfg, "game_objects", None) or getattr(getattr(cfg, "game", None), "objects", None) or {}


def _summary_vector(
    cfg: MettaGridConfig,
    *,
    resource_types: Sequence[str],
    sentinels: Tuple[str, str] = ("nothing", "heart"),
    max_assemblers: int = 12,
) -> np.ndarray:
    """
    Construct a lossy assembler summary suitable as auxiliary model input.

    This block is NOT used when reconstructing `MettaGridConfig` and its exact
    layout may evolve over time. It is intentionally non-invertible and should
    be treated purely as contextual signal for representation learning
    (e.g. VAE inputs, auxiliary predictors), not as part of the core reversible
    config schema.

    Current:

      1. resource_histogram (len(resource_types)):
         Normalized frequency of each resource that appears along a simple
         "main chain" (start -> ... -> goal), excluding the goal sentinel.
         If no matching resources are found, this slice is all zeros.

      2. num_assemblers_norm (1):
         Total number of assemblers, clipped at `max_assemblers` and scaled
         into [0, 1].

      3. unique_resource_fraction (1):
         Fraction of distinct resources used along the main chain (excluding
         the goal sentinel) relative to len(resource_types), in [0, 1].

    If to be extended in the future, (e.g. cooldown statistics, branching factors),
    append new scalars at the end and update helpers that depend on the size
    of this block accordingly.
    """
    objs = _objects(cfg)

    num_assemblers = sum(1 for o in objs.values() if getattr(o, "type", None) == "assembler")
    num_assemblers_norm = float(min(max_assemblers, max(0, num_assemblers))) / float(max_assemblers)

    start, goal = sentinels

    next_of: dict[str, str] = {}
    for o in objs.values():
        if getattr(o, "type", None) != "assembler":
            continue

        protos = getattr(o, "protocols", []) or []
        if not protos:
            continue

        p = protos[0]
        out = dict(getattr(p, "output_resources", {}) or {})
        inp = dict(getattr(p, "input_resources", {}) or {})

        if len(out) != 1:
            continue

        out_res = next(iter(out))

        if len(inp) == 0:
            in_res = start
        elif len(inp) == 1:
            in_res = next(iter(inp))
        else:
            continue

        next_of[in_res] = out_res

    chain_outputs: List[str] = []
    cur = start
    for _ in range(64):
        if cur not in next_of:
            break
        out_res = next_of[cur]
        chain_outputs.append(out_res)
        if out_res == goal:
            break
        cur = out_res

    effective_outputs = [r for r in chain_outputs if r != goal]

    idx = {name: i for i, name in enumerate(resource_types)}
    hist: np.ndarray = np.zeros(len(resource_types), dtype=np.float32)
    for r in effective_outputs:
        j = idx.get(r)
        if j is not None:
            hist[j] += 1.0

    total = float(hist.sum())
    if total > 0.0:
        hist /= total

    if effective_outputs and len(resource_types) > 0:
        unique_resource_fraction = min(
            1.0,
            float(len(set(effective_outputs))) / float(len(resource_types)),
        )
    else:
        unique_resource_fraction = 0.0

    return np.concatenate(
        [
            hist,
            np.array(
                [num_assemblers_norm, float(unique_resource_fraction)],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )


def serialize_payload(
    config: MettaGridConfig,
    *,
    resource_types: Sequence[str] | None = None,
    sentinels: Tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Extended payload that includes a *non-invertible* assembler summary."""
    base_blocks = serialize_config(config)
    base_vec = np.concatenate([base_blocks["continuous"], base_blocks["categorical"]], axis=0)

    payload: dict[str, Any] = {
        "meta": {
            "resource_types": list(resource_types) if resource_types else None,
            "sentinels": None,
        },
        "base": base_vec,
        "summary": None,
    }
    if resource_types:
        sents = sentinels if sentinels is not None else ("nothing", "heart")
        payload["meta"]["sentinels"] = {"start": sents[0], "goal": sents[1]}
        payload["summary"] = _summary_vector(config, resource_types=resource_types, sentinels=sents)
    return payload


def flatten_payload(payload: dict[str, Any], *, include_summary: bool = True) -> np.ndarray:
    """
    Flatten payload into a single feature vector.

    If include_summary is True and a summary is present:
        concat(base, summary)
    Otherwise:
        return base

    Suitable as direct input to representation models.
    """
    base = np.asarray(payload["base"], dtype=np.float32)
    if include_summary and payload.get("summary") is not None:
        return np.concatenate([base, np.asarray(payload["summary"], dtype=np.float32)], axis=0)
    return base
