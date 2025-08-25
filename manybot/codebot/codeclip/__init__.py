# Re-export codeclip's public API to avoid deep imports elsewhere
from .codeclip.file import get_context  # noqa: F401

__all__ = ["get_context"]
