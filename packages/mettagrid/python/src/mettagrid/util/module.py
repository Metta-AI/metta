import importlib
import logging

logger = logging.getLogger(__name__)


def load_symbol(full_name: str):
    """Load a symbol from a full name, for example: 'mettagrid.base_config.Config' -> Config."""
    remaining_name = full_name
    symbol_path: list[str] = []

    # Keep stripping off the last part until the remaining name is a valid module name.
    #
    # This handles the case where the symbol name is a nested class, e.g.
    # 'mettagrid.map_builder.ascii.AsciiMapBuilder.Config'
    while True:
        parts = remaining_name.rsplit(".", 1)
        if len(parts) != 2:
            raise ModuleNotFoundError(f"Invalid symbol name: {full_name}")

        remaining_name, symbol_name = parts
        symbol_path.insert(0, symbol_name)
        try:
            logger.debug(f"Loading module {remaining_name} with symbol path {symbol_path}")
            module = importlib.import_module(remaining_name)
            value = module
            for symbol_name in symbol_path:
                value = getattr(value, symbol_name)
            return value
        except ImportError:
            pass
