"""Utility modules for metta."""

# Lazy imports - modules are imported when accessed, not at package load time
# This avoids pulling in heavy dependencies (like boto3) when only using basic CLI features

__all__ = ["file"]


def __getattr__(name: str):
    """Lazily import submodules when accessed.

    This allows `from metta.utils import file` to work while avoiding eager imports.
    """
    if name == "file":
        import importlib

        return importlib.import_module("mettagrid.util.file")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
