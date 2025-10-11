"""Utilities for working with the MettaGrid action catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from mettagrid.core import MettaGridCore


def build_action_mapping(env: "MettaGridCore") -> Tuple[Dict[int, Tuple[int, int]], List[str]]:
    """Return mapping from flattened action index to (action_id, action_param) and base action names.

    This prefers the canonical catalog exposed by the C++ bindings and falls back to legacy heuristics
    when running against an older engine.
    """

    mapping: Dict[int, Tuple[int, int]] = {}
    base_names: List[str] = []

    c_env = getattr(env, "c_env", None)
    if c_env is not None and hasattr(c_env, "action_catalog"):
        catalog_entries = list(c_env.action_catalog())
        max_index = -1
        for entry in catalog_entries:
            action_id = int(entry["action_id"])
            flat_index = int(entry["flat_index"])
            param = int(entry["param"])
            mapping[flat_index] = (action_id, param)
            if action_id > max_index:
                max_index = action_id

        if mapping and max_index >= 0:
            base_names = ["" for _ in range(max_index + 1)]
            for entry in catalog_entries:
                action_id = int(entry["action_id"])
                if base_names[action_id] == "":
                    base_names[action_id] = str(entry["base_name"])
            if all(name != "" for name in base_names):
                return mapping, base_names

    # Fallback: derive mapping using flat names and max args (older bindings).
    flat_names = list(env.action_names)
    if not flat_names:
        return mapping, base_names

    max_args: List[int] | None = None
    if c_env is not None and hasattr(c_env, "max_action_args"):
        try:
            max_args = list(c_env.max_action_args())
        except Exception:  # pragma: no cover - protective
            max_args = None

    if max_args:
        cursor = 0
        for base_id, max_arg in enumerate(max_args):
            count = max(1, max_arg + 1)
            if cursor >= len(flat_names):
                break
            raw_name = flat_names[cursor]
            base_name = raw_name.rsplit("_", 1)[0] if max_arg > 0 and "_" in raw_name else raw_name
            base_names.append(base_name)
            for param in range(count):
                if cursor >= len(flat_names):
                    break
                mapping[cursor] = (base_id, param if max_arg > 0 else 0)
                cursor += 1

        while cursor < len(flat_names):
            base_id = len(base_names)
            base_names.append(flat_names[cursor])
            mapping[cursor] = (base_id, 0)
            cursor += 1
    else:
        for idx, name in enumerate(flat_names):
            mapping[idx] = (idx, 0)
            base_names.append(name)

    return mapping, base_names


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
