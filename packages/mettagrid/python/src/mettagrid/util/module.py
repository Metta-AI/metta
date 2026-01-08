import importlib


def load_symbol(full_name: str, strict: bool = True):
    """Load a symbol from a full name, for example: 'mettagrid.base_config.Config' -> Config.

    Handles nested attributes like 'mettagrid.map_builder.ascii.AsciiMapBuilder.Config'.
    """
    parts = full_name.split(".")
    if len(parts) < 2:
        raise ModuleNotFoundError(f"Invalid symbol name: {full_name}")

    # Try importing progressively shorter module paths until one works
    last_error = None
    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            last_error = e
            continue
        try:
            # Navigate through the remaining attributes
            value = module
            for attr_name in parts[i:]:
                value = getattr(value, attr_name)
            return value
        except AttributeError:
            continue

    # If we get here, we couldn't load the symbol
    if strict:
        if last_error:
            raise last_error
        raise ModuleNotFoundError(f"Could not load symbol: {full_name}")
    return None
