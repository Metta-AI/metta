"""Array utility functions for Metta training."""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

ReadableFmt = Literal["json", "yaml"]


def _infer_fmt(path: Path, fmt: ReadableFmt | None) -> ReadableFmt:
    if fmt in ("json", "yaml"):
        return fmt
    suf = path.suffix.lower()
    if suf in {".yml", ".yaml"}:
        return "yaml"
    return "json"


def save_3d_array_readable(
    path: str | Path,
    data: Any,
    *,
    fmt: ReadableFmt | None = None,
    round_fp: int | None = None,
) -> Path:
    """Save a 3D array to a *human‑readable* text file (JSON or YAML).

    - Preserves shape & dtype
    - Uses nested lists for readability (depth -> rows -> cols)
    - Optionally rounds floating‑point values for smaller/cleaner files

    Parameters
    ----------
    path : str | Path
        Output file path. If no extension is given, defaults to .json.
    data : Any
        Array-like object convertible to a NumPy array.
    fmt : {"json", "yaml"} | None
        Force output format. If None, inferred from the file extension.
    round_fp : int | None
        If provided, round floats to this many decimals before saving.

    Returns
    -------
    Path
        The resolved output path.
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.ndim}D")

    p = Path(path)
    chosen_fmt = _infer_fmt(p, fmt)
    if p.suffix == "":
        p = p.with_suffix(".json" if chosen_fmt == "json" else ".yaml")

    # Convert to nested lists for readability
    nested: Any = arr.tolist()

    if round_fp is not None:
        # Recursively round floats inside nested lists
        def _round(v: Any) -> Any:
            if isinstance(v, float):
                return round(v, round_fp)
            if isinstance(v, list):
                return [_round(x) for x in v]
            return v

        nested = _round(nested)

    payload = {
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": nested,
    }

    if chosen_fmt == "json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    else:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required for YAML output. Install with `pip install pyyaml` or use fmt='json'."
            ) from e
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return p.resolve()


# --- Example ---
# x = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4)
# save_3d_array_readable("array.yaml", x, fmt="yaml", round_fp=None)
# x2 = load_3d_array_readable("array.yaml")
# assert np.array_equal(x, x2)
