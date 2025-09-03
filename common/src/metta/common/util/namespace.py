import pkgutil

from metta.common.util.log_config import init_logging


def setup_metta_namespace_package(name: str, path: list[str]) -> list[str]:
    """
    Set up a namespace package with proper logging configuration.

    Usage in __init__.py:
        from metta.common.util.namespace import setup_namespace_package
        __path__ = setup_namespace_package(__name__)
    """
    # Extend path for namespace package
    extended_path = pkgutil.extend_path(path, name)
    init_logging()

    return list(extended_path)
