"""Registry system for block types."""

from __future__ import annotations

from typing import Callable, Dict, Type

from cortex.blocks.base import BaseBlock
from cortex.cells.base import MemoryCell
from cortex.config import BlockConfig

# Type for block builder functions
BlockBuilder = Callable[[BlockConfig, int, MemoryCell], BaseBlock]

# Global registry of block types
_BLOCK_REGISTRY: Dict[Type[BlockConfig], Type[BaseBlock]] = {}


def register_block(config_class: Type[BlockConfig]) -> Callable:
    """Decorator to register a block class with its config type.

    Usage:
        @register_block(MyBlockConfig)
        class MyBlock(BaseBlock):
            ...
    """

    def decorator(block_class: Type[BaseBlock]) -> Type[BaseBlock]:
        _BLOCK_REGISTRY[config_class] = block_class
        # Also store a reference in the config class for convenience
        config_class._block_class = block_class
        return block_class

    return decorator


def get_block_class(config: BlockConfig) -> Type[BaseBlock]:
    """Get the block class for a given config instance."""
    config_type = type(config)

    # First check if config has _block_class attribute
    if hasattr(config_type, "_block_class"):
        return config_type._block_class

    # Fall back to registry
    if config_type in _BLOCK_REGISTRY:
        return _BLOCK_REGISTRY[config_type]

    raise ValueError(f"No block class registered for config type {config_type.__name__}")


def build_block(config: BlockConfig, d_hidden: int, cell: MemoryCell) -> BaseBlock:
    """Build a block from its configuration.

    This is the generic builder that works for any registered block type.
    """
    block_class = get_block_class(config)
    return block_class(config=config, d_hidden=d_hidden, cell=cell)


__all__ = ["register_block", "get_block_class", "build_block"]
