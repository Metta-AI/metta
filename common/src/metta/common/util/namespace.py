import logging
import sys

from metta.common.util.log_config import init_logging


def setup_namespace_package(name: str) -> list[str]:
    """
    Set up a namespace package with proper logging configuration.

    This function:
    1. Extends the module path for namespace packages
    2. Calls init_logging() (which is idempotent due to @functools.cache)
    3. Adds a NullHandler to prevent "No handler" warnings

    Usage in __init__.py:
        from metta.common.util.namespace import setup_namespace_package
        __path__ = setup_namespace_package(__name__)
    """
    # Get the calling module
    module = sys.modules[name]

    # Extend path for namespace package
    import pkgutil

    extended_path = pkgutil.extend_path(module.__path__, name)

    # Initialize logging (idempotent due to @functools.cache decorator)
    init_logging()

    # Add NullHandler to prevent "No handler" warnings for this package
    logger = logging.getLogger(name)
    # Only add if no handlers exist to avoid duplicates
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return extended_path
