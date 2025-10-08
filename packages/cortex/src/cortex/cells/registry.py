"""Registry for memory cells."""

from typing import Callable, Type

from cortex.cells.base import MemoryCell
from cortex.config import CellConfig

# Global registry mapping config classes to cell classes
_CELL_REGISTRY: dict[Type[CellConfig], Type[MemoryCell]] = {}


def register_cell(config_class: Type[CellConfig]) -> Callable:
    """Register decorator linking cell class to its configuration type."""

    def decorator(cell_class: Type[MemoryCell]) -> Type[MemoryCell]:
        _CELL_REGISTRY[config_class] = cell_class
        # Also store reverse mapping on config class for convenience
        config_class._cell_class = cell_class  # type: ignore
        return cell_class

    return decorator


def get_cell_class(config: CellConfig) -> Type[MemoryCell]:
    """Lookup cell class from configuration instance."""
    config_type = type(config)
    if config_type not in _CELL_REGISTRY:
        raise ValueError(f"No cell registered for config type {config_type.__name__}")
    return _CELL_REGISTRY[config_type]


def build_cell(config: CellConfig) -> MemoryCell:
    """Instantiate cell from configuration using registry lookup."""
    cell_class = get_cell_class(config)
    return cell_class(config)


__all__ = ["register_cell", "build_cell", "get_cell_class"]
