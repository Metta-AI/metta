from __future__ import annotations

from typing import Any, Mapping

from mettagrid.util.module import load_symbol


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    import ast

    spec = spec.strip()
    if not spec:
        raise ValueError("architecture_spec cannot be empty")

    expr = ast.parse(spec, mode="eval").body

    if isinstance(expr, ast.Call):
        class_path = _expr_to_dotted(expr.func)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg}
    elif isinstance(expr, (ast.Name, ast.Attribute)):
        class_path = _expr_to_dotted(expr)
        kwargs = {}
    else:
        raise ValueError("Unsupported policy architecture specification format")

    return class_path, kwargs


def format_spec(class_path: str, payload: Mapping[str, Any]) -> str:
    class_path = class_path.strip()
    if not class_path:
        raise ValueError("architecture_spec cannot be empty")

    if not payload:
        return class_path

    sorted_payload = _sorted_structure(dict(payload))
    parts = [f"{key}={repr(sorted_payload[key])}" for key in sorted(sorted_payload)]
    return f"{class_path}({', '.join(parts)})"


def architecture_from_spec(spec: str) -> Any:
    class_path, _ = parse_spec(spec)
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
    spec = spec.strip()
    if not spec:
        raise ValueError("architecture_spec cannot be empty")
    return spec


def _sorted_structure(value: Any) -> Any:
    from collections.abc import Mapping as MappingABC

    if isinstance(value, MappingABC):
        return {key: _sorted_structure(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_structure(item) for item in value]
    return value


def _expr_to_dotted(expr) -> str:
    import ast

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{_expr_to_dotted(expr.value)}.{expr.attr}"
    raise ValueError("Expected a dotted name for policy architecture class path")
