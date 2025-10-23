"""Registry for memory cells."""

from typing import Callable, Type

from cortex.cells.base import MemoryCell
from cortex.config import CellConfig

# Global registry mapping config classes to cell classes
_CELL_REGISTRY: dict[Type[CellConfig], Type[MemoryCell]] = {}

# Tag -> config class mapping to support easy extensibility and JSON round‑trip
_CELL_CONFIG_BY_TAG: dict[str, Type[CellConfig]] = {}


def _get_cell_tag(config_class: type[CellConfig]) -> str:
    """Return the stable tag declared on the config class.

    We expect Pydantic v2 `model_fields` to contain a default for `cell_type`.
    """
    field = config_class.model_fields["cell_type"]  # type: ignore[attr-defined]
    tag = field.default  # type: ignore[assignment]
    if isinstance(tag, str) and tag:
        return tag
    raise ValueError(f"Cell config {config_class.__name__} must define a default 'cell_type' field")


def register_cell(config_class: Type[CellConfig]) -> Callable:
    """Register decorator linking cell class to its configuration type."""

    def decorator(cell_class: Type[MemoryCell]) -> Type[MemoryCell]:
        _CELL_REGISTRY[config_class] = cell_class
        # Also store reverse mapping on config class for convenience
        config_class._cell_class = cell_class  # type: ignore[attr-defined]

        # Register the configuration tag → class mapping for round‑trip parsing
        tag = _get_cell_tag(config_class)
        if tag in _CELL_CONFIG_BY_TAG and _CELL_CONFIG_BY_TAG[tag] is not config_class:
            raise ValueError(
                f"Duplicate cell_type tag '{tag}' for {config_class.__name__};"
                f" already registered to {_CELL_CONFIG_BY_TAG[tag].__name__}"
            )
        _CELL_CONFIG_BY_TAG[tag] = config_class
        return cell_class

    return decorator


def get_cell_class(config: CellConfig) -> Type[MemoryCell]:
    """Lookup cell class from configuration instance."""
    config_type = type(config)
    if config_type not in _CELL_REGISTRY:
        raise ValueError(f"No cell registered for config type {config_type.__name__}")
    return _CELL_REGISTRY[config_type]


def get_cell_config_class(tag: str) -> type[CellConfig]:
    """Return the CellConfig subclass registered for a given tag."""
    if tag not in _CELL_CONFIG_BY_TAG:
        raise KeyError(f"Unknown cell_type tag '{tag}' — is the cell registered?")
    return _CELL_CONFIG_BY_TAG[tag]


def build_cell(config: CellConfig) -> MemoryCell:
    """Instantiate cell from configuration using registry lookup."""
    cell_class = get_cell_class(config)
    return cell_class(config)


__all__ = [
    "register_cell",
    "build_cell",
    "get_cell_class",
    "get_cell_config_class",
]
