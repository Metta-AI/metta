"""Utility modules for metta."""

# Lazy imports - modules are imported when accessed, not at package load time
# This avoids pulling in heavy dependencies (like boto3) when only using basic CLI features

__all__ = ["file", "uri"]
