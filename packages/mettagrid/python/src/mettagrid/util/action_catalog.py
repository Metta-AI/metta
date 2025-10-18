"""Utilities for working with the MettaGrid action catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from mettagrid.core import MettaGridCore


def build_action_mapping(env: "MettaGridCore") -> Tuple[Dict[int, Tuple[int, int]], List[str]]:
    """Return mapping from flattened action index to (action_id, action_param) plus base action names."""

    c_env = getattr(env, "c_env", None)
    if c_env is None or not hasattr(c_env, "action_catalog"):
        raise RuntimeError("MettaGrid environment does not expose action_catalog; rebuild bindings.")

    catalog_entries = list(c_env.action_catalog())
    if not catalog_entries:
        raise ValueError("MettaGrid action catalog is empty; cannot build action mapping.")

    mapping: Dict[int, Tuple[int, int]] = {}
    base_names: Dict[int, str] = {}

    for entry in catalog_entries:
        flat_index = int(entry["flat_index"])
        action_id = int(entry["action_id"])
        param = int(entry["param"])
        mapping[flat_index] = (action_id, param)
        base_name = str(entry["base_name"])
        base_names.setdefault(action_id, base_name)

    max_action_id = max(base_names.keys())
    ordered_base_names: List[str] = [""] * (max_action_id + 1)
    for action_id, name in base_names.items():
        ordered_base_names[action_id] = name

    if "" in ordered_base_names:
        missing = [idx for idx, name in enumerate(ordered_base_names) if name == ""]
        raise ValueError(f"Missing base action names for ids: {missing}")

    return mapping, ordered_base_names


def make_decode_fn(mapping: Mapping[int, Tuple[int, int]]) -> Callable[[int], Tuple[int, int]]:
    """Return a callable that converts flattened indices using the provided mapping."""

    def decode(flat_index: int) -> Tuple[int, int]:
        return mapping[flat_index]

    return decode


def make_encode_fn(mapping: Mapping[int, Tuple[int, int]]) -> Callable[[int, int], int]:
    """Return a callable that converts (action_id, action_param) into flattened indices."""

    inverse: Dict[Tuple[int, int], int] = {pair: flat_index for flat_index, pair in mapping.items()}

    def encode(action_id: int, action_param: int) -> int:
        return inverse[(int(action_id), int(action_param))]

    return encode
