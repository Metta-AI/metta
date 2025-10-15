import importlib


def load_symbol(full_name: str):
    """Load a symbol from a full name, for example: 'mettagrid.base_config.Config' -> Config."""
    parts = full_name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol name: {full_name}")
    module_name, symbol_name = parts
    module = importlib.import_module(module_name)
    value = getattr(module, symbol_name)
    return value
