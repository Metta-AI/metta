from __future__ import annotations

from typing import Any

from mettagrid.util.module import load_symbol


def architecture_from_spec(spec: str) -> Any:
    spec = spec.strip()
    if not spec:
        raise ValueError("architecture_spec cannot be empty")

    class_path = spec.split("(")[0].strip()
    config_class = load_symbol(class_path)
    if not isinstance(config_class, type):
        raise TypeError(f"Loaded symbol {class_path} is not a class")
    if not hasattr(config_class, "from_spec"):
        raise TypeError(f"Class {class_path} does not have a from_spec method")
    return config_class.from_spec(spec)


def architecture_spec_from_value(architecture: Any) -> str:
    if isinstance(architecture, str):
        spec = architecture
    elif hasattr(architecture, "to_spec"):
        spec = architecture.to_spec()
    else:
        raise TypeError("architecture must be a spec string or provide to_spec()")
    if not spec.strip():
        raise ValueError("architecture_spec cannot be empty")
    return spec.strip()
