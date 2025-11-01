"""Datadog monitoring and dashboard management."""

# Lazy import to avoid requiring CLI dependencies for programmatic use
__all__ = ["datadog_app"]


def __getattr__(name):
    """Lazy import for CLI app."""
    if name == "datadog_app":
        from devops.datadog.cli import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
