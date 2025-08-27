# Re-export codeclip's public API to avoid deep imports elsewhere
from .codeclip.file import CodeContext, Document, get_context, get_context_objects  # noqa: F401

__all__ = ["get_context", "get_context_objects", "Document", "CodeContext"]
