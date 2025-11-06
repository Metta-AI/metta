"""Registry system for block types."""


import typing

import cortex.blocks.base
import cortex.cells.base
import cortex.config

# Type for block builder functions
BlockBuilder = typing.Callable[
    [cortex.config.BlockConfig, int, cortex.cells.base.MemoryCell], cortex.blocks.base.BaseBlock
]

# Global registry of block types
_BLOCK_REGISTRY: typing.Dict[typing.Type[cortex.config.BlockConfig], typing.Type[cortex.blocks.base.BaseBlock]] = {}

# Tag -> config class mapping for robust JSON round‑trip
_BLOCK_CONFIG_BY_TAG: typing.Dict[str, type[cortex.config.BlockConfig]] = {}


def _get_block_tag(config_class: type[cortex.config.BlockConfig]) -> str:
    field = config_class.model_fields["block_type"]  # type: ignore[attr-defined]
    tag = field.default  # type: ignore[assignment]
    if isinstance(tag, str) and tag:
        return tag
    raise ValueError(f"Block config {config_class.__name__} must define a default 'block_type' field")


def register_block(config_class: typing.Type[cortex.config.BlockConfig]) -> typing.Callable:
    """Register decorator linking block class to its configuration type."""

    def decorator(block_class: typing.Type[cortex.blocks.base.BaseBlock]) -> typing.Type[cortex.blocks.base.BaseBlock]:
        _BLOCK_REGISTRY[config_class] = block_class
        # Also store a reference in the config class for convenience
        config_class._block_class = block_class  # type: ignore[attr-defined]

        # Register tag mapping
        tag = _get_block_tag(config_class)
        if tag in _BLOCK_CONFIG_BY_TAG and _BLOCK_CONFIG_BY_TAG[tag] is not config_class:
            raise ValueError(
                f"Duplicate block_type tag '{tag}' for {config_class.__name__};"
                f" already registered to {_BLOCK_CONFIG_BY_TAG[tag].__name__}"
            )
        _BLOCK_CONFIG_BY_TAG[tag] = config_class
        return block_class

    return decorator


def get_block_class(config: cortex.config.BlockConfig) -> typing.Type[cortex.blocks.base.BaseBlock]:
    """Lookup block class from configuration instance."""
    config_type = type(config)

    # First check if config has _block_class attribute
    if hasattr(config_type, "_block_class"):
        return config_type._block_class

    # Fall back to registry
    if config_type in _BLOCK_REGISTRY:
        return _BLOCK_REGISTRY[config_type]

    raise ValueError(f"No block class registered for config type {config_type.__name__}")


def get_block_config_class(tag: str) -> type[cortex.config.BlockConfig]:
    if tag not in _BLOCK_CONFIG_BY_TAG:
        raise KeyError(f"Unknown block_type tag '{tag}' — is the block registered?")
    return _BLOCK_CONFIG_BY_TAG[tag]


def build_block(
    config: cortex.config.BlockConfig, d_hidden: int, cell: cortex.cells.base.MemoryCell
) -> cortex.blocks.base.BaseBlock:
    """Instantiate block from configuration using registry lookup."""
    block_class = get_block_class(config)
    return block_class(config=config, d_hidden=d_hidden, cell=cell)


__all__ = ["register_block", "get_block_class", "build_block", "get_block_config_class"]
