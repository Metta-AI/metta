import importlib


def load_symbol(full_name: str):
    """Load a symbol from a full name, for example: 'mettagrid.base_config.Config' -> Config."""
    remaining_name = full_name
    symbol_path: list[str] = []

    # Keep stripping off the last part until we get to a symbol name that doesn't start with an uppercase letter.
    #
    # This handles the case where the symbol name is a nested class, e.g.
    # 'mettagrid.map_builder.ascii.AsciiMapBuilder.Config'
    while True:
        parts = remaining_name.rsplit(".", 1)
        if len(parts) != 2:
            break
        next_module_name, next_symbol_name = parts

        if not symbol_path or next_symbol_name[0].isupper():
            # accepting the split if either:
            # 1. we haven't split yet (symbol_path is empty)
            # 2. the next symbol name starts with an uppercase letter
            remaining_name = next_module_name
            symbol_path.insert(0, next_symbol_name)
        else:
            break

    if len(symbol_path) == 0:
        raise ValueError(f"Invalid symbol name: {full_name}")

    module = importlib.import_module(remaining_name)
    value = module
    for symbol_name in symbol_path:
        value = getattr(value, symbol_name)
    return value
